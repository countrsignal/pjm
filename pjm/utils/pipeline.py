from dgl.dataloading import GraphDataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import torch

import wandb
from pytorch_lightning import seed_everything

from ..model.training_utils import UnitTest, _setup_logger, load_model
from .noam_opt import get_std_opt, Adafactor
from .data import  Collator, AF2SCN
from .tokenizer import Alphabet

from multiprocessing import current_process
from datetime import datetime
from typing import Any, Optional
from numpy import mean
from tqdm import tqdm
import logging
import os


__all__ = ['setup_ddp', 'cleanup', 'launch_ddp', 'Pipeline']


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def launch_ddp(training_fn, world_size, model_args, dataloader_args, unit_test_callback, early_stop_callback):
    mp.spawn(training_fn,
             args=(world_size, model_args, dataloader_args, unit_test_callback, early_stop_callback),
             nprocs=world_size,
             join=True)


def _unit_test_enabled(unit_test):
    return issubclass(unit_test.__class__, UnitTest)


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


def _early_stop_enabled(early_stop):
    return issubclass(early_stop.__class__, EarlyStopping)


class Pipeline(object):

    def __init__(
            self,
            training_args: Any,
            parallel_training: bool = False,
            ):

        super().__init__()
        self.config = training_args
        self.num_epochs = training_args.num_epochs
        self.alphabet = Alphabet(
            prepend_toks = ("<sos>", "<eos>", "<unk>"),
            append_toks = ("<cls>", "<pad>", "<mask>"),
            prepend_bos = True,
            append_eos = True,
            use_msa = False
        )
        self.scaler = GradScaler()
        self.dataset_path = training_args.dataset_path
        self.distributed = parallel_training

    def training_loader(self, **kwargs):
        collate_fn = Collator(tokenizer=self.alphabet.get_batch_converter())
        train_set = AF2SCN(
                split='train',
                max_len=self.config.max_len,
                dataset_path=self.dataset_path,
                _filter_by_plddt_coverage=self.config.plddt_filter
                )
        # << ! >> Unit test hook
        if _unit_test_enabled(self.unit_test):
            self.unit_test.msg("Dataset override triggered")
            train_set = self.unit_test.get_unit_test_dataset(dataset=train_set)
        # << ! >> Early stop hook
        if _early_stop_enabled(self.early_stop):
            self.early_stop.msg("Dataset override triggered")
            train_set = self.early_stop.get_early_stop_dataset(dataset=train_set)

        if self.distributed:
            loader = GraphDataLoader(
                    train_set,
                    use_ddp=True,
                    collate_fn=collate_fn,
                    shuffle=True,
                    **kwargs
                    )
        else:
            loader = DataLoader(
                    train_set,
                    collate_fn=collate_fn,
                    shuffle=True,
                    **kwargs
            )
        return loader

    def test_loader(self, **kwargs):
        collate_fn = Collator(tokenizer=self.alphabet.get_batch_converter())
        test = AF2SCN(split='test', max_len=1022, dataset_path=self.dataset_path)
        if self.distributed:
            loader = GraphDataLoader(
                    test,
                    use_ddp=True,
                    collate_fn=collate_fn,
                    shuffle=False,
                    **kwargs
                    )
        else:
            loader = DataLoader(
                    test,
                    collate_fn=collate_fn,
                    shuffle=False,
                    **kwargs
            )
        return loader

    def train_step(self, epoch_index, batch_index, batch, model, optimizer):
        
        if self.distributed:
            device = model.module.encoder_parallel_device
        else:
            if model.encoder_parallel_device is not None:
                device = model.encoder_parallel_device
            else:
                device = 'cuda:0'
        
        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            sequences, *structures = batch.process_data(device)

            if (epoch_index == 0) and (batch_index == 0):
                if self.distributed:
                    model.module.dispatch_params()
                else:
                    if model.encoder_parallel_device:
                        model.dispatch_params()
                    else:
                        model = model.to(device)

            cont_loss, ar_loss, total_loss = model(
                sequences=sequences,
                structures=structures,
                return_embeddings=False,
                return_loss=True
                )

        # Scale loss. Backward pass under autocast is not recommended.
        self.scaler.scale(total_loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        return cont_loss.detach().cpu().item(), ar_loss.detach().cpu().item()

    def evaluate(self, model, **kwargs):

        if self.distributed:
            device = model.module.encoder_parallel_device
        else:
            device = 'cuda:0'

        # Place model in eval mode
        model.eval()
        with torch.no_grad():
            losses = []
            for batch in self.test_loader(**kwargs):
                # Forward pass with autocast
                with autocast():
                    sequences, *structures = batch.process_data(device)
                    cont_loss, ar_loss, total_loss = model.forward(
                        sequences=sequences,
                        structures=structures,
                        return_embeddings=False,
                        return_loss=True
                        )
                losses.append(total_loss.detach().cpu().item())
        # ( ! ) Return model to training mode
        model.train()
        return mean(losses)

    def fit(
        self,
        rank,
        world_size,
        model_args,
        dataloader_args,
        unit_test_callback: Optional[UnitTest] = None,
        early_stop_callback: Optional[EarlyStopping] = None,
    ):

        # Set random seed
        seed_everything(self.config.seed)
        
        # Setup wandb logger
        if not self.distributed:
            print("Training without distributed training")
            run = wandb.init(
                dir=".",
                config=vars(self.config),
                project="jessy",
                tags=self.config.tags
            )
            run_dir = run.dir
        else:
            if rank == 0:
                print(f"Dristributed training enabled.\n")
                print(f"World size is {world_size}")
                run = wandb.init(
                    dir=".",
                    config=vars(self.config),
                    project="jessy",
                    tags=self.config.tags
                )
                run_dir = run.dir

        ############################
        # Distributed Data Parallel
        ############################
        if self.distributed:
            print(f"This message is from {current_process().name}")
            # Setup DDP
            setup_ddp(rank, world_size)
            
            # Setup unit test callback
            setattr(self, "unit_test", unit_test_callback)
            if _unit_test_enabled(self.unit_test):
                self.unit_test.msg("Unit test initialized")
            # Set up early stopping callback
            setattr(self, "early_stop", early_stop_callback)
            if _early_stop_enabled(self.early_stop):
                self.early_stop.msg("Early stopping initialized")
            
            # ( ! ) Set model to distributed mode
            dev0 = (rank * 2) % (world_size + 2)
            dev1 = (rank * 2 + 1) % (world_size + 2)
            # > Distributed model
            # << ! >> Unit test hook
            if _unit_test_enabled(self.unit_test):
                model = self.unit_test.get_unit_test_model(
                    devices=(dev0, dev1),
                    alphabet=self.alphabet,
                    model_args=model_args
                )
            else:
                model = load_model((dev0, dev1), self.alphabet, model_args)
            model = DDP(model, broadcast_buffers=False)
            # > Optimizer
            opt = get_std_opt(
                model.parameters(),
                d_model=model_args['dim'],
                factor=0.98 if not _unit_test_enabled(self.unit_test) else 0.1,
                warmup=int(8653 * 4) if not _unit_test_enabled(self.unit_test) else 10,
                weight_decay=model_args['weight_decay']
            )

            # > Watch gradients only for rank 0
            if rank == 0:
               run.watch(model, log="all", log_freq=self.config.log_interval)
        ############################
        #       Model Parallel
        ############################
        else:
            # Setup unit test callback
            setattr(self, "unit_test", unit_test_callback)
            if _unit_test_enabled(self.unit_test):
                self.unit_test.msg("Unit test initialized")
            
            # Enable model parallelism if there are 2 GPUs
            # Otherwise, use single GPU
            if torch.cuda.device_count() == 2:
                dev0 = torch.device('cuda:0')
                dev1 = torch.device('cuda:1')
            else:
                dev0 = None
                dev1 = None
            # ( ! ) Set model to non-distributed mode
            # Setup unit test callback
            if _unit_test_enabled(self.unit_test):
                model = self.unit_test.get_unit_test_model(
                    devices=(dev0, dev1),
                    alphabet=self.alphabet,
                    model_args=model_args
                )
            else:
                model = load_model((dev0, dev1), self.alphabet, model_args)
            # > Watch model parameters and gradients
            run.watch(model, log="all", log_freq=self.config.log_interval)
            # Set up early stopping callback
            setattr(self, "early_stop", early_stop_callback)
            if _early_stop_enabled(self.early_stop):
                self.early_stop.msg("Early stopping initialized")
            
            # > Optimizer
            opt = get_std_opt(
                model.parameters(),
                d_model=model_args['dim'],
                factor=0.98 if not _unit_test_enabled(self.unit_test) else 0.1,
                warmup=int(8653 * 4) if not _unit_test_enabled(self.unit_test) else 100,
                weight_decay=model_args['weight_decay']
            )
            # opt = Adafactor(
            #     model.parameters(),
            #     lr=None,
            #     weight_decay=model_args['weight_decay'],
            #     warmup_init=True
            # )
            # opt = torch.optim.Adam(
            #     model.parameters(),
            #     lr=model_args['lr'],
            #     betas=(0.9, 0.98),
            #     eps=1e-9,
            #     weight_decay=model_args['weight_decay']
            # )

        # ( ! ) Model Training
        best_eval = 1e6
        for epoch_index in range(self.num_epochs):
            # > Initialize training data loader
            train_loader = self.training_loader(**dataloader_args)

            # > Initialize TQDM progress bar
            if self.distributed:
                # ( ! ) DGL Distributed DataLoader
                # > We need to repeat the random partition every epoch to guarantee randomness
                train_loader.set_epoch(epoch_index)

                training_progress_bar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Rank {rank}: Epoch 1, Loss (NA)",
                    mininterval=1 if _unit_test_enabled(self.unit_test) else 50,
                    position=rank
                    )
            else:
                training_progress_bar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch 1, Loss (NA)",
                    mininterval=1 if _unit_test_enabled(self.unit_test) else 50,
                    )
            
            # > Early stopping optimizer hook
            if _early_stop_enabled(self.early_stop):
                if epoch_index == self.early_stop.patience:
                    self.early_stop.update_optimizer(opt.optimizer)

            # > Epoch Loop
            for batch_index, batch in training_progress_bar:

                losses = self.train_step(epoch_index, batch_index, batch, model, opt)

                # > Logging training loss
                if not self.distributed:
                    if (batch_index + 1) % self.config.log_interval == 0:
                        with torch.no_grad():
                            dict_norm = model.structure_queries['queries'].norm().item() if hasattr(model, "structure_queries") else 0
                            node_proj_norm = model.structure_encoder[0].n_proj.ffn[0].wv.weight.norm().item()
                            seq_enc_attn_norm = model.sequence_encoder[0].layers[0].fn.to_qkv.weight.norm().item()
                            seq_dec_attn_norm = model.decoder.stack[0].layers[0].layers[0].fn.to_qkv.weight.norm().item()
                            seq_dec_logit_layer_norm = model.decoder.stack[-1].weight.norm().item()

                        run.log({
                            "Learning Rate": opt.param_groups[0]['lr'],
                            "Training Loss": sum(losses),
                            "Contrastive Loss": losses[0],
                            "Autoregressive Loss": losses[1],
                            "Tempurature": model.temperature['temperature'].item(),
                            "Structure Query Norm": dict_norm,
                            "Structure Node Projection Norm": node_proj_norm,
                            "Sequence Encoder Attention Norm": seq_enc_attn_norm,
                            "Sequence Decoder Attention Norm": seq_dec_attn_norm,
                            "Sequence Decoder Logit Layer Norm": seq_dec_logit_layer_norm,
                        })

                    if (batch_index + 1) % training_progress_bar.mininterval == 0:
                        training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)})")
                else:
                    if rank == 0:
                        if (batch_index + 1) % self.config.log_interval == 0:

                            with torch.no_grad():
                                dict_norm = model.module.structure_queries['queries'].norm().item() if hasattr(model.module, "structure_queries") else 0
                                node_proj_norm = model.module.structure_encoder[0].n_proj.ffn[0].wv.weight.norm().item()
                                seq_enc_attn_norm = model.module.sequence_encoder[0].layers[0].fn.to_qkv.weight.norm().item()
                                seq_dec_attn_norm = model.module.decoder.stack[0].layers[0].layers[0].fn.to_qkv.weight.norm().item()
                                seq_dec_logit_layer_norm = model.module.decoder.stack[-1].weight.norm().item()

                            run.log({
                                "Learning Rate": opt.param_groups[0]['lr'],
                                "Training Loss": sum(losses),
                                "Contrastive Loss": losses[0],
                                "Autoregressive Loss": losses[1],
                                "Tempurature": model.module.temperature['temperature'].item(),
                                "Structure Query Norm": dict_norm,
                                "Structure Node Projection Norm": node_proj_norm,
                                "Sequence Encoder Attention Norm": seq_enc_attn_norm,
                                "Sequence Decoder Attention Norm": seq_dec_attn_norm,
                                "Sequence Decoder Logit Layer Norm": seq_dec_logit_layer_norm,
                            })

                    if (batch_index + 1) % training_progress_bar.mininterval == 0:
                        training_progress_bar.set_description(f"Rank {rank}: Epoch {epoch_index + 1}, Loss ({sum(losses)})")

            # > Evaluate on test set
            loss = self.evaluate(model, **dataloader_args)
            if loss < best_eval:
                best_eval = loss
                timestamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                if not self.distributed:
                    print(f"Validation on Epoch {epoch_index}: {loss}")
                    checkpoint_state = {
                            'epoch': epoch_index,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict()
                            }
                    torch.save(
                            checkpoint_state,
                            os.path.join(run_dir, f'model_chkpt_{timestamp}.pth')
                            )
                else:
                    if rank == 0:
                        print(f"Validation on Epoch {epoch_index}: {loss}")
                        checkpoint_state = {
                                'epoch': epoch_index,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': opt.state_dict()
                                }
                        torch.save(
                                checkpoint_state,
                                os.path.join(run_dir, f'model_chkpt_{timestamp}.pth')
                                )

        # > Close out run
        if self.distributed:
            # > Close logger
            if rank == 0:
                wandb.finish()
            # > Close DDP
            cleanup()
        else:
            wandb.finish()
