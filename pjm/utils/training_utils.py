from typing import Optional, Union, Tuple, Dict
from multiprocessing import current_process
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from inspect import getfullargspec
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import logging
import random
import yaml
import sys
import os

import wandb
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from ..tokenizer import Alphabet
from ..data import Collator, AF2SCN
from ..model.multimodal import MMPLM
from ..model.baseline import BaselineModel
from ..model.experimental import CoCa
from ..model.gvp_gnn import standard_structure_module


_MM_LOSS_LOG_ = [
    "Training Loss",
    "CLIP Loss",
    "Masked Residue Loss",
    "Node MSE Loss",
    "Distogram Loss (Alphas)",
    "Distogram Loss (Betas)",
]

_SPLITS_ = ["training", "validation", "test"]


def set_training_env_vars():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training_parser():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="Set the random seed.")
    parser.add_argument("--config_path", type=str, help="Set path to config file.")
    parser.add_argument("--gpu_devices", nargs="+", help="Set devices to use for training.")
    parser.add_argument("--multi_modal", action="store_true", help="Set to train multi-modal model.")
    parser.add_argument("--epochs", type=int, help="Set number of epochs to train.")
    parser.add_argument("--log_interval", type=int, default=200, help="Set logging frequency.")
    parser.add_argument("--val_interval", type=int, default=2000, help="Set logging frequency.")
    parser.add_argument("--run_name", type=str, help="Name for the experiment.")
    parser.add_argument("--tags", nargs="+", help="W&B run tags.")
    parser.add_argument("--debug", action="store_true", help="Set to aid in debugging.")
    parser.add_argument("--detect_anomaly", action="store_true", help="Set to aid in debugging.")
    return parser.parse_args()


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


def launch_ddp(training_fn, world_size, model_args, dataloader_args, unit_test_callback, early_stop_callback):
    mp.spawn(training_fn,
             args=(world_size, model_args, dataloader_args, unit_test_callback, early_stop_callback),
             nprocs=world_size,
             join=True)


def cleanup():
    dist.destroy_process_group()


def _update_config_(config, key, value):
    config.update({key: value})


def _override_config_(config, key, value):
    if key in config:
        config[key] = value
    else:
        raise KeyError(f"Key {key} not found in config.")


def build_model(config, alphabet, multi_modal=True, **kwargs):
    if multi_modal:
        model =  MMPLM(config=config, alphabet=alphabet, **kwargs)
    else:
        model = BaselineModel(
            alphabet=alphabet,
            **config,
        )
    return model


def build_optimizer(model, config):
    if config["optimizer"]["type"].lower() == "adam":
        opt = torch.optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    elif config["optimizer"]["type"].lower() == "adamw":
        opt = torch.optim.AdamW(
            params=model.parameters(),
            lr=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']['type']} not recognized.")
    if config["optimizer"]["lr_scheduler"] is not None:
        lr_scheduler = WarmupLinearSchedule(optimizer=opt, **config["optimizer"]["lr_scheduler"])
    else:
        lr_scheduler = None
    return opt, lr_scheduler


def build_default_alphabet():
    return Alphabet(
        prepend_toks = ("<sos>", "<eos>", "<unk>"),
        append_toks = ("<cls>", "<pad>", "<mask>"),
        prepend_bos = True,
        append_eos = True,
        use_msa = False
    )


def assembler(config_path: str, multi_modal: bool = True):
    # Load config yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load alphabet
    alphabet = build_default_alphabet()
    # Initialize collator
    model_type = "mmplm" if multi_modal else "baseline"
    collate_fn = Collator(config=config['model'][model_type]['data']['collate'], tokenizer=alphabet.get_batch_converter())
    _update_config_(config['model'][model_type]['data']['loader'], 'collate_fn', collate_fn)
    # Initialize training dataloaders
    train_loader = DataLoader(
        dataset=AF2SCN(**config['model'][model_type]['data']['training']),
        shuffle=True,
        **config['model'][model_type]['data']['loader']
    )

    # Initialize model
    model = build_model(
        config=config['model'][model_type]['architecture'],
        alphabet=alphabet,
        multi_modal=multi_modal,
    )
    return config, train_loader, model


def forward_pass_hook(multi_modal, model, *args, **kwargs):
    if torch.cuda.is_bf16_supported():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            losses = model(*args, **kwargs)
            if multi_modal:
                total_loss = sum(losses)
    else:
        losses = model(*args, **kwargs)
        if multi_modal:
            total_loss = sum(losses)
    
    if multi_modal:
        return total_loss, *losses
    else:
        return (losses, )


def backward_pass_hook(loss, optimizer, scheduler):
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def training_step_hook(debug, multi_modal, model, optimizer, scheduler, *args, **kwargs):
    if debug:
        with torch.autograd.detect_anomaly():
            losses = forward_pass_hook(multi_modal, model, *args, **kwargs)
            backward_pass_hook(losses[0], optimizer, scheduler)
    else:
        losses = forward_pass_hook(multi_modal, model, *args, **kwargs)
        backward_pass_hook(losses[0], optimizer, scheduler)
    return losses


def logging_hook(multi_modal, losses, lr_scheduler):
    log_dict = {}
    if multi_modal:
        for idx, l, in enumerate(losses):
            log_dict[_MM_LOSS_LOG_[idx]] = l.detach().cpu().item()
    else:
        log_dict["Masked Residue Loss"] = losses[0].detach().cpu().item()
    
    if lr_scheduler is not None:
        log_dict["Learning Rate"] = lr_scheduler.get_last_lr()[0]
    return log_dict


def init_progress_bar(epoch_index, train_loss, val_loss, log_interval, train_loader):
    return tqdm(
        enumerate(train_loader),
        desc=f"Epoch {epoch_index + 1} | Training Step Loss ({train_loss}) | Best Validation Loss ({val_loss})",
        total=len(train_loader),
        mininterval=log_interval,
    )


def init_runner(run_name, tags, config):
    runner = wandb.init(
        dir=".",
        name=run_name,
        project="protein-language-modeling",
        tags=tags,
        config=config,
    )
    return runner


def write_to_logger(runner, log_dict):
    runner.log(log_dict)


def validation_step_hook(multi_modal, val_loader, model, *args, **kwargs):
    # Average loss across all batches
    if multi_modal:
        validation_loss = {"Val " + l: [] for l in _MM_LOSS_LOG_[1:]}
        validation_loss["Validation Loss"] = []
    else:
        validation_loss = {"Validation Loss": []}
    # Enter evaluation mode
    model.eval()
    # Identify model device
    if multi_modal:
        # NOTE: for multi-modal PLM, sequences AND structures are sent to decoder module first
        input_device = model.decoder_parallel_device
    else:
        # NOTE: for baseline PLM, sequences are sent to encoder module first
        input_device = model.encoder_parallel_device
    for batch_index, batch in enumerate(val_loader):
        seuqence, *structure = batch.process_data(input_device, multi_modal)
        with torch.no_grad():
            if multi_modal:
                losses = forward_pass_hook(True, model, seuqence, structure, *args, **kwargs)
            else:
                losses = forward_pass_hook(False, model, seuqence, *args, **kwargs)
        if multi_modal:
            for idx, l in enumerate(losses):
                if idx == 0:
                    validation_loss["Validation Loss"].append(l.detach().cpu().item())
                else:
                    validation_loss["Val " + _MM_LOSS_LOG_[idx]].append(l.detach().cpu().item())
        else:
            validation_loss["Validation Loss"].append(losses[0].detach().cpu().item())
    # Return model to training mode
    model.train()
    # Return average loss across all batches
    return {k: np.mean(v) for k, v in validation_loss.items()}


def write_checkpoint(model, optimizer, scheduler, epoch, config, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "config": config,
    }
    torch.save(checkpoint, path)


def load_jem(
    devices: Tuple[Union[int, str]],
    alphabet: Alphabet,
    model_args: Dict[str, Union[int, str, float, bool]],
    _sanity_check_mode_: bool = False,
    ) -> nn.Module:
    
    dev0, dev1 = devices
    config = {
        "dim": model_args["embedding_dim"],
        "alphabet": alphabet,
        "num_transformer_blocks": model_args["num_sequence_transformer_blocks"],
        "contrastive_loss_weight": model_args["contrastive_loss_weight"],
        "cross_entropy_loss_weight": model_args["cross_entropy_loss_weight"],
        "cross_exchange_decoding": model_args["cross_exchange_decoding"],
        "corrupt_structure": model_args["corrupt_structure"],
        "structure_reconstruction": model_args["structure_reconstruction"],
        "sturcture_global_projection": model_args["sturcture_global_projection"],
        'encoder_parallel_device':dev0,
        'decoder_parallel_device': dev1,
        'depth': model_args['transformer_block_depth'],
        'heads': model_args['num_attns_heads'],
        'head_dim': model_args['attn_head_dim'],
        'dropout': model_args["dropout"],
    }

    # Structure encoder args
    if not _sanity_check_mode_:
        strc_enc_args = {}
        strc_enc_args = {
            "node_in_dims": model_args["node_in_dims"],
            "node_out_dims": model_args["node_out_dims"],
            "edge_in_dims": model_args["edge_in_dims"],
            "num_edge_gvps": model_args["num_edge_gvps"],
            "num_gvp_convs": model_args["num_gvp_convs"],
            "final_proj_dim": model_args["embedding_dim"],
            "num_transformer_blocks": model_args["num_structure_transformer_blocks"],
            "transformer_input_dim": model_args["embedding_dim"],
            "transformer_block_depth": model_args["transformer_block_depth"],
            "num_attns_heads": model_args["num_attns_heads"],
            "attn_head_dim": model_args["attn_head_dim"],
            "dropout": model_args["dropout"],
        }
        config['structure_encoder'] = standard_structure_module(**strc_enc_args)
    else:
        config['structure_encoder'] = None

    return CoCa(**config)


def load_model(
    devices: Tuple[Union[int, str]],
    alphabet: Alphabet,
    model_args: Dict[str, Union[int, str, float, bool, nn.Module]],
    ) -> nn.Module:
    dev0, dev1 = devices
    config = {
        'encoder_parallel_device':dev0,
        'decoder_parallel_device': dev1,
        'depth': model_args['depth'],
        'heads': model_args['heads'],
        'dim_head': model_args['dim_head'],
        'dropout': 0.1
    }

    strc_enc_args = {}
    for argument in getfullargspec(standard_structure_module).args:
        strc_enc_args[argument] = model_args[argument]
    config['structure_encoder'] = standard_structure_module(**strc_enc_args)

    # config['alphabet_size'] = len(alphabet.all_toks)
    # config['padding_index'] = alphabet.padding_idx

    if model_args["model_type"].lower() == "coca":
        for argument in getfullargspec(CoCa.__init__).args[1:-2]:
            # if argument not in ('structure_encoder', 'alphabet_size', 'padding_index'):
            if argument != 'structure_encoder':
                # if argument == 'mask_index':
                #     config[argument] = alphabet.mask_idx
                # elif argument == 'sos_token_index':
                #     config[argument] = alphabet.sos_idx
                # elif argument == 'eos_token_index':
                #     config[argument] = alphabet.eos_idx
                # elif argument == 'cls_token_index':
                #     config[argument] = alphabet.cls_idx
                if argument == 'alphabet':
                    config[argument] = alphabet
                else:
                    config[argument] = model_args[argument]
            else:
                continue
        return CoCa(**config)

    else:
        raise NotImplementedError(f"Model type {model_args['model_type']} not implemented")


def _setup_logger(logger_name: str, log_level: int):
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(log_level)

    # create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _early_stop_enabled(early_stop):
    return issubclass(early_stop.__class__, EarlyStopping)


def _unit_test_enabled(unit_test):
    return issubclass(unit_test.__class__, UnitTest)


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps`
        steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(
            max(1.0, self.t_total - self.warmup_steps)))


class UnitTest(ABC):

    def __init__(self):
        self.logger = _setup_logger(logger_name=str(self), log_level=logging.DEBUG)

    def __str__(self) -> str:
        name = self.__class__.__name__
        if name == 'UnitTest':
            name = 'Generic'
        return f"UnitTest(type={name})"

    def __repr__(self) -> str:
        return self.__str__()

    def msg(self, message: str):
        process_name = current_process().name
        if (process_name == 'MainProcess') or (process_name == 'SpawnProcess-1'):
            self.logger.debug(message)

    @staticmethod
    @abstractmethod
    def get_unit_test_dataset(dataset: AF2SCN) -> AF2SCN:
        pass


class Overfit(UnitTest):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_unit_test_dataset(dataset: AF2SCN, num_samples: int) -> AF2SCN:
        new_manifest = {}
        af2_count, scn_count = 0, 0
        max_per_data_source = num_samples // 2
        for k, v in dataset.manifest.items():
            if len(v['sequence']) <= dataset.max_len:
                if k.split('-')[0] == 'AF':
                    if af2_count < max_per_data_source:
                        new_manifest[k] = v
                        af2_count += 1
                    else:
                        continue
                else:
                    if scn_count < max_per_data_source:
                        new_manifest[k] = v
                        scn_count += 1
                    else:
                        continue
            else:
                continue
        dataset._overwrite_manifest(new_manifest=new_manifest)
        return dataset

    def get_unit_test_model(self, alphabet, devices, model_args):
        self.msg("Unit test model hook")
        self.msg(f"Loading model {model_args['model_type']}")
        return load_model(devices, alphabet, model_args)


class EarlyStopping(object):
    def __init__(self, patience: int = 5, reduce_factor: float = 0.005, reduce_ratio: float = 0.5):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.reduce_factor = reduce_factor
        self.reduce_ratio = reduce_ratio
        self.logger = _setup_logger(logger_name=str(self), log_level=logging.INFO)
    
    def __str__(self):
        status = "IN-PROGRESS" if not self.early_stop else "STOPPED"
        return f"EarlyStopping(patience={self.patience}, counter={self.counter}, status={status})"
    
    def __repr__(self):
        return self.__str__()
    
    def msg(self, message: str):
        process_name = current_process().name
        if (process_name == 'MainProcess') or (process_name == 'SpawnProcess-1'):
            self.logger.info(message)

    def get_early_stop_dataset(self, dataset: AF2SCN) -> AF2SCN:
        if self.counter < self.patience:
            self.counter += 1
            num_examples = int(len(dataset) * self.reduce_factor)
            amount = int(num_examples / self.reduce_ratio)
            new_manifest = {}
            af2_count, scn_count = 0, 0
            for k, v in dataset.manifest.items():
                if k.split('-')[0] == 'AF':
                    if af2_count < amount:
                        new_manifest[k] = v
                        af2_count += 1
                    else:
                        continue
                else:
                    if scn_count < amount:
                        new_manifest[k] = v
                        scn_count += 1
                    else:
                        continue
            dataset._overwrite_manifest(new_manifest=new_manifest)
        else:
            self.early_stop = True
        
        return dataset
    
    def update_optimizer(self, optimizer: torch.optim.Optimizer):
        if self.early_stop:
            optimizer.param_groups[0]['weight_decay'] = 0.0
            self.msg(f"Setting weight decay to 0.0")


class EvalMonitor(object):
    def __init__(self, split: str, multi_modal: bool, monitor_interval: int, config: dict):
        super().__init__()
        assert split in _SPLITS_[1:], f"Split {split} not recognized."
        self.split = split
        self.multi_modal = multi_modal
        self.monitor_interval = monitor_interval
        self.loader = DataLoader(
            dataset=AF2SCN(**config['data'][split]),
            shuffle=False,
            **config['data']['loader']
        )
        self.best_eval_loss = float("inf")
        self.experiment_config = deepcopy(config)
    
    def __str__(self):
        return f"EvalMonitor(split={self.split}, multi_modal={self.multi_modal}, monitor_interval={self.monitor_interval})"
    
    def __repr__(self):
        return self.__str__()
    
    def watch(self, global_step):
        if (global_step % self.monitor_interval) == 0:
            return True
        else:
            return False
    
    def evaluation_step(self, epoch_index, runner, model, optimizer, scheduler, ckpt_model=True):
        if self.split == "validation":
            eval_log_dict = validation_step_hook(self.multi_modal, self.loader, model)
            eval_loss = eval_log_dict["Validation Loss"]
        elif self.split == "test":
            raise NotImplementedError("Test split not implemented.")
        else:
            raise ValueError(f"Split {self.split} not recognized.")
        
        runner.log(eval_log_dict)
        if ckpt_model:
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                write_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch_index + 1,
                    self.experiment_config,
                    os.path.join(runner.dir, f"{runner.name}_ckpt.pth"),
                )