from dgl.dataloading import GraphDataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import torch

import wandb
from pytorch_lightning import seed_everything

from .training_utils import UnitTest, _setup_logger, load_model
from ..model.baseline import BaselineModel
from .data import  Collator, AF2SCN
from .tokenizer import Alphabet


from multiprocessing import current_process
from datetime import datetime
from typing import Any, Optional
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
        self.dataset_path = training_args.dataset_path
        self.distributed = parallel_training

    def training_loader(self, **kwargs):
        # Training dataset
        collate_fn = Collator(tokenizer=self.alphabet.get_batch_converter())
        train_set = AF2SCN(
                split='train',
                max_len=self.config.max_len,
                dataset_path=self.dataset_path,
                _sequence_only_baseline=False if self.config.multimodal else True,
                _filter_by_plddt_coverage=self.config.plddt_filter
                )
        
        # Callback hooks
        # << ! >> Unit test hook
        if _unit_test_enabled(self.unit_test):
            self.unit_test.msg("Dataset override triggered")
            train_set = self.unit_test.get_unit_test_dataset(dataset=train_set)
        # << ! >> Early stop hook
        if _early_stop_enabled(self.early_stop):
            self.early_stop.msg("Dataset override triggered")
            train_set = self.early_stop.get_early_stop_dataset(dataset=train_set)

        # Init dataloaders
        if (self.distributed) and (self.config.multimodal):
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

    def val_loader(self, **kwargs):
        # Validation dataset
        collate_fn = Collator(tokenizer=self.alphabet.get_batch_converter())
        val = AF2SCN(
            split='val',
            max_len=1022,
            dataset_path=self.dataset_path,
            _sequence_only_baseline=False if self.config.multimodal else True,
            _filter_by_plddt_coverage=None,
            )
        
        # Init dataloaders
        if (self.distributed) and (self.config.multimodal):
            loader = GraphDataLoader(
                    val,
                    use_ddp=True,
                    collate_fn=collate_fn,
                    shuffle=False,
                    **kwargs
                    )
        else:
            loader = DataLoader(
                    val,
                    collate_fn=collate_fn,
                    shuffle=False,
                    **kwargs
            )
        return loader

    def train_step(self, epoch_index, batch_index, batch, model, optimizer, scheduler=None):
        
        if self.distributed:
            device = model.module.encoder_parallel_device
        else:
            if model.encoder_parallel_device is not None:
                device = model.encoder_parallel_device
            else:
                device = 'cuda:0'
        
        optimizer.zero_grad()

        # Forward pass with autocast
        # > BFLOAT16 does not require GradScaler!
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.config.bfloat16):
            
            # >> Move data (and model) to device(s)
            if self.config.multimodal:
                sequences, *structures = batch.process_data(device)
            else:
                sequences = batch.seqs.to(device)

            if (epoch_index == 0) and (batch_index == 0):
                if self.distributed:
                    model.module.dispatch_params()
                else:
                    if model.encoder_parallel_device:
                        model.dispatch_params()
                    else:
                        model = model.to(device)

            # >> Forward pass
            if self.config.multimodal:
                cont_loss, ce_loss, total_loss = model(
                    sequences=sequences,
                    structures=structures,
                    return_embeddings=False,
                    return_loss=True
                    )
            else:
                ce_loss = model(sequences)

        # Backward pass (under autocast is not recommended)
        if self.config.multimodal:
            total_loss.backward()
        else:
            ce_loss.backward()
        
        # Optimizer step
        optimizer.step()
        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        if self.config.multimodal:
            return [cont_loss.detach().cpu().item(), ce_loss.detach().cpu().item()]
        else:
            return [ce_loss.detach().cpu().item(),]

    def evaluate(self, model, **kwargs):

        if self.distributed:
            device = model.module.encoder_parallel_device
        else:
            device = 'cuda:0'

        # Place model in eval mode
        model.eval()
        with torch.no_grad():
            losses = []
            for batch in self.val_loader(**kwargs):
                # Forward pass with autocast
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.config.bfloat16):
                    if self.config.multimodal:
                        sequences, *structures = batch.process_data(device)
                        *_, total_loss = model.forward(
                            sequences=sequences,
                            structures=structures,
                            return_embeddings=False,
                            return_loss=True
                            )
                    else:
                        sequences = batch.seqs.to(device)
                        total_loss = model(sequences)

                losses.append(total_loss.detach().cpu().item())
        # ( ! ) Return model to training mode
        model.train()
        return sum(losses) / len(losses)

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
                project="joint embeddings",
                name="jess" if self.config.multimodal else "baseline",
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
                    project="joint embeddings",
                    name="jess" if self.config.multimodal else "baseline",
                    tags=self.config.tags,
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
                # << ! >> Multi-modal or Uni-modal model
                if self.config.multimodal:
                    model = load_model((dev0, dev1), self.alphabet, model_args)
                else:
                    transformer_config = {
                        "depth": model_args["depth"],
                        "heads": model_args["heads"],
                        "dim_head": model_args["dim_heads"],
                        "dropout": model_args["dropout"],
                    }
                    model = BaselineModel(
                        dim=model_args["dim_heads"],
                        alphabet=self.alphabet,
                        num_layers=model_args["num_layers"],
                        encoder_parallel_device=dev0,
                        decoder_parallel_device=dev1,
                        **transformer_config,
                    )
            model = DDP(model, broadcast_buffers=False)

            # > Optimizer
            if self.config.multimodal:
                opt = torch.optim.Adam(
                    model.parameters(),
                    lr=model_args["lr"],
                    betas=(0.9, 0.98),
                    eps=1e-9,
                )
                lr_scheduler = None
            else:
                opt = torch.optim.AdamW(
                    model.parameters(),
                    lr=model_args["lr"],
                    betas=(0.9, 0.98),
                    eps=1e-9,
                    weight_decay=model_args["weight_decay"],
                    )
                if model_args["lr_scheduler"]:
                    lr_scheduler = WarmupLinearSchedule(optimizer=opt, **model_args["lr_scheduler"])
                else:
                    lr_scheduler = None

            # > Watch gradients only for rank 0
            # if rank == 0:
            #    run.watch(model, log="all", log_freq=self.config.log_interval)

        ############################
        #       Model Parallel
        ############################
        else:
            # Setup unit test callback
            setattr(self, "unit_test", unit_test_callback)
            if _unit_test_enabled(self.unit_test):
                self.unit_test.msg("Unit test initialized")
            # Set up early stopping callback
            setattr(self, "early_stop", early_stop_callback)
            if _early_stop_enabled(self.early_stop):
                self.early_stop.msg("Early stopping initialized")
            
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
               # << ! >> Multi-modal or Uni-modal model
                if self.config.multimodal:
                    model = load_model((dev0, dev1), self.alphabet, model_args)
                else:
                    transformer_config = {
                        "depth": model_args["depth"],
                        "heads": model_args["heads"],
                        "dim_head": model_args["dim_heads"],
                        "dropout": model_args["dropout"],
                    }
                    model = BaselineModel(
                        dim=model_args["dim"],
                        alphabet=self.alphabet,
                        num_layers=model_args["num_layers"],
                        encoder_parallel_device=dev0,
                        decoder_parallel_device=dev1,
                        **transformer_config,
                    )
            
            # > Optimizer
            if self.config.multimodal:
                opt = torch.optim.Adam(
                    model.parameters(),
                    lr=model_args["lr"],
                    betas=(0.9, 0.98),
                    eps=1e-9,
                )
                lr_scheduler = None
            else:
                opt = torch.optim.AdamW(
                    model.parameters(),
                    lr=model_args["lr"],
                    betas=(0.9, 0.98),
                    eps=1e-9,
                    weight_decay=model_args["weight_decay"],
                    )
                if model_args["lr_scheduler"]:
                    lr_scheduler = WarmupLinearSchedule(optimizer=opt, **model_args["lr_scheduler"])
                else:
                    lr_scheduler = None

        # ( ! ) Model Training
        global_step = 0
        losses = [None, None]
        best_eval = float("inf")
        timestamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        for epoch_index in range(self.num_epochs):
            # > Initialize training data loader
            train_loader = self.training_loader(**dataloader_args)

            # > Initialize TQDM progress bar
            if self.distributed:
                # ( ! ) DGL Distributed DataLoader ( ONLY FOR MULTI-MODAL TRAINING )
                # > We need to repeat the random partition every epoch to guarantee randomness
                if self.config.multimodal:
                    train_loader.set_epoch(epoch_index)

                if rank == 0:
                    if all([l is None for l in losses]):
                        loss_msg = "NA"
                    else:
                        loss_msg = sum(losses)
                    
                    if best_eval == float("inf"):
                        eval_msg = "NA"
                    else:
                        eval_msg = best_eval
                    
                    training_progress_bar = tqdm(
                        enumerate(train_loader),
                        total=len(train_loader),
                        desc=f"Rank {rank}: Epoch {epoch_index + 1}, Loss ({loss_msg}), Best Val Loss: ({eval_msg})",
                        mininterval=1 if _unit_test_enabled(self.unit_test) else self.config.log_interval,
                        position=rank
                        )
                else:
                    training_progress_bar = enumerate(train_loader)

            else:

                if all([l is None for l in losses]):
                    loss_msg = "NA"
                else:
                    loss_msg = sum(losses)
                
                if best_eval == float("inf"):
                    eval_msg = "NA"
                else:
                    eval_msg = best_eval
                
                training_progress_bar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch_index + 1}, Loss ({loss_msg}), Best Val Loss: ({eval_msg})",
                    mininterval=1 if _unit_test_enabled(self.unit_test) else self.config.log_interval,
                    )
            
            # > Early stopping optimizer hook
            if _early_stop_enabled(self.early_stop):
                if epoch_index == self.early_stop.patience:
                    self.early_stop.update_optimizer(opt.optimizer)

            # > Epoch Loop
            for batch_index, batch in training_progress_bar:

                losses = self.train_step(epoch_index, batch_index, batch, model, opt, lr_scheduler)
                global_step += 1

                # Skip Validation if unit test is enabled
                if _unit_test_enabled(self.unit_test):
                    # > Only update TQDM progress bar and log to wandb
                    if (self.distributed) and (self.config.multimodal):
                        if rank == 0:
                            losses[0] = losses[0] / model_args["contrastive_loss_weight"]
                            losses[1] = losses[1] / model_args["cross_entropy_loss_weight"]
                            run.log({
                                "Learning Rate": opt.param_groups[0]['lr'],
                                "Total Loss": sum(losses),
                                "Contrastive Loss": losses[0],
                                "Cross-Entropy Loss": losses[1],
                            })
                    elif (not self.distributed) and (self.config.multimodal):
                        losses[0] = losses[0] / model_args["contrastive_loss_weight"]
                        losses[1] = losses[1] / model_args["cross_entropy_loss_weight"]
                        run.log({
                            "Learning Rate": opt.param_groups[0]['lr'],
                            "Total Loss": sum(losses),
                            "Contrastive Loss": losses[0],
                            "Cross-Entropy Loss": losses[1],
                        })
                    elif (self.distributed) and (not self.config.multimodal):
                        if rank == 0:
                            run.log({
                                "Learning Rate": opt.param_groups[0]['lr'],
                                "Cross-Entropy Loss": losses[0],
                            })
                    else:
                        run.log({
                            "Learning Rate": opt.param_groups[0]['lr'],
                            "Cross-Entropy Loss": losses[0],
                        })
                    
                    # Update TQDM progress bar
                    training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: (NA)")

                else:
                    # > Logging training and validation loss
                    if (self.distributed) and (self.config.multimodal):
                        if rank == 0:
                            if global_step % self.config.log_interval == 0:
                                # Evaluate on validation set
                                val_loss = self.evaluate(model, **dataloader_args)
                                if best_eval > val_loss:
                                    best_eval = val_loss
                                    checkpoint_state = {
                                            'epoch': epoch_index,
                                            'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': opt.state_dict()
                                            }
                                    torch.save(
                                            checkpoint_state,
                                            os.path.join(run_dir, f'model_chkpt_{timestamp}.pth')
                                            )

                                # Re-weight multi-modal losses
                                losses[0] = losses[0] / model_args["contrastive_loss_weight"]
                                losses[1] = losses[1] / model_args["cross_entropy_loss_weight"]

                                # Update TQDM progress bar
                                training_progress_bar.set_description(f"Rank {rank}: Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: ({best_eval})")

                                # Log to wandb
                                run.log({
                                    "Learning Rate": opt.param_groups[0]['lr'],
                                    "Total Loss": sum(losses),
                                    "Contrastive Loss": losses[0],
                                    "Cross-Entropy Loss": losses[1],
                                    "Validation Set Loss": val_loss,
                                })

                    elif (not self.distributed) and (self.config.multimodal):
                        if global_step % self.config.log_interval == 0:
                            # Evaluate on validation set
                            val_loss = self.evaluate(model, **dataloader_args)
                            if best_eval > val_loss:
                                best_eval = val_loss
                                checkpoint_state = {
                                        'epoch': epoch_index,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': opt.state_dict()
                                        }
                                torch.save(
                                        checkpoint_state,
                                        os.path.join(run_dir, f'model_chkpt_{timestamp}.pth')
                                        )
                            
                            # Re-weight multi-modal losses
                            losses[0] = losses[0] / model_args["contrastive_loss_weight"]
                            losses[1] = losses[1] / model_args["cross_entropy_loss_weight"]

                            # Update TQDM progress bar
                            training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: ({best_eval})")

                            # Log to wandb
                            run.log({
                                "Learning Rate": opt.param_groups[0]['lr'],
                                "Total Loss": sum(losses),
                                "Contrastive Loss": losses[0],
                                "Cross-Entropy Loss": losses[1],
                                "Validation Set Loss": val_loss,
                            })

                    elif (self.distributed) and (not self.config.multimodal):
                        if rank == 0:
                            if global_step % self.config.log_interval == 0:
                                # Evaluate on validation set
                                val_loss = self.evaluate(model, **dataloader_args)
                                if best_eval > val_loss:
                                    best_eval = val_loss
                                    checkpoint_state = {
                                            'epoch': epoch_index,
                                            'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': opt.state_dict()
                                            }
                                    torch.save(
                                            checkpoint_state,
                                            os.path.join(run_dir, f'model_chkpt_{timestamp}.pth')
                                            )

                                # Update TQDM progress bar
                                training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: ({best_eval})")

                                # Log to wandb
                                run.log({
                                    "Learning Rate": opt.param_groups[0]['lr'],
                                    "Cross-Entropy Loss": losses[0],
                                    "Validation Set Loss": val_loss,
                                })

                    else:
                        if global_step % self.config.log_interval == 0:
                            # Evaluate on validation set
                            val_loss = self.evaluate(model, **dataloader_args)
                            if best_eval > val_loss:
                                best_eval = val_loss
                                checkpoint_state = {
                                        'epoch': epoch_index,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': opt.state_dict()
                                        }
                                torch.save(
                                        checkpoint_state,
                                        os.path.join(run_dir, f'model_chkpt_{timestamp}.pth')
                                        )
                            
                            # Update TQDM progress bar
                            training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: ({best_eval})")

                            # Log to wandb
                            run.log({
                                "Learning Rate": opt.param_groups[0]['lr'],
                                "Cross-Entropy Loss": losses[0],
                                "Validation Set Loss": val_loss,
                            })

        # > Close out run
        if self.distributed:
            # > Close logger
            if rank == 0:
                wandb.finish()
            # > Close DDP
            cleanup()
        else:
            wandb.finish()
