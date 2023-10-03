import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from pytorch_lightning import seed_everything

from ..tokenizer import Alphabet
from ..data import  Collator, AF2SCN
from ..model.baseline import BaselineModel
from .training_utils import (
    WarmupLinearSchedule,
    EarlyStopping,
    UnitTest,
    setup_ddp,
    cleanup,
    _unit_test_enabled,
    _early_stop_enabled,
    load_jem,
    _MM_LOSS_LOG_,
)


from multiprocessing import current_process
from datetime import datetime
from typing import Any, Optional
from tqdm import tqdm
import os


__all__ = ['Pipeline']


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
            train_set = self.unit_test.get_unit_test_dataset(
                dataset=train_set,
                num_samples=self.config.batch_size,
                )
        # << ! >> Early stop hook
        if _early_stop_enabled(self.early_stop):
            self.early_stop.msg("Dataset override triggered")
            train_set = self.early_stop.get_early_stop_dataset(dataset=train_set)

        # Init dataloaders
        if (self.distributed):
            loader = DataLoader(
                    train_set,
                    shuffle=False,
                    collate_fn=collate_fn,
                    sampler=DistributedSampler(train_set),
                    **kwargs
                    )
        else:
            loader = DataLoader(
                    train_set,
                    shuffle=True,
                    collate_fn=collate_fn,
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
        
        # Init dataloader
        loader = DataLoader(
                val,
                shuffle=False,
                collate_fn=collate_fn,
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

        # Zero gradients        
        optimizer.zero_grad()

        # Move data and model to device(s)
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

        # Forward pass with autocast
        # > BFLOAT16 does not require GradScaler!
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.config.bfloat16):

            # >> Forward pass
            if self.config.multimodal:
                losses = model(
                    sequences=sequences,
                    structures=structures,
                    return_embeddings=False,
                    return_loss=True
                    )
            else:
                ce_loss = model(sequences)

        # Backward pass (under autocast is not recommended)
        if self.config.multimodal:
            total_loss = losses[0]
            total_loss.backward()
        else:
            ce_loss.backward()
        
        # Optimizer step
        optimizer.step()
        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        if self.config.multimodal:
            return [l.detach().cpu().item() for l in losses]
        else:
            return [ce_loss.detach().cpu().item(),]

    def evaluate(self, validation_loader, model):

        if self.distributed:
            device = model.module.encoder_parallel_device
        else:
            device = 'cuda:0'

        # Place model in eval mode
        model.eval()
        with torch.no_grad():
            losses = []
            for batch in validation_loader:
                # Forward pass with autocast
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.config.bfloat16):
                    if self.config.multimodal:
                        sequences, *structures = batch.process_data(device)
                        losses = model.forward(
                            sequences=sequences,
                            structures=structures,
                            )
                        total_loss = losses[0]
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
                name=f"jess-{self.config.name}" if self.config.multimodal else f"baseline-{self.config.name}",
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
                    name=f"jess-{self.config.name}" if self.config.multimodal else f"baseline-{self.config.name}",
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
                    model = load_jem((dev0, dev1), self.alphabet, model_args)
                    if (rank == 0):
                        print('Trainable Params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
                else:
                    transformer_config = {
                        "heads": model_args["num_attns_heads"],
                        "head_dim": model_args["attn_head_dim"],
                        "dropout": model_args["dropout"],
                    }
                    model = BaselineModel(
                        dim=model_args["embedding_dim"],
                        alphabet=self.alphabet,
                        num_attn_layers=model_args["num_attn_layers"],
                        encoder_parallel_device=dev0,
                        decoder_parallel_device=dev1,
                        **transformer_config,
                    )
                    if (rank == 0):
                        print('Trainable Params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
            model = DDP(model, broadcast_buffers=False)

            # > Optimizer
            if self.config.multimodal:
                if self.config.weight_decay:
                    opt = torch.optim.AdamW(
                        model.parameters(),
                        lr=model_args["lr"],
                        betas=(0.9, 0.95),
                        eps=1e-9,
                        weight_decay=model_args["weight_decay"],
                        )
                else:
                    opt = torch.optim.Adam(
                        model.parameters(),
                        lr=model_args["lr"],
                        betas=(0.9, 0.95),
                        eps=1e-9,
                    )

                if model_args["lr_scheduler"]:
                    lr_scheduler = WarmupLinearSchedule(optimizer=opt, **model_args["lr_scheduler"])
                else:
                    lr_scheduler = None

            else:
                if self.config.weight_decay:
                    opt = torch.optim.AdamW(
                        model.parameters(),
                        lr=model_args["lr"],
                        betas=(0.9, 0.95),
                        eps=1e-9,
                        weight_decay=model_args["weight_decay"],
                        )
                else:
                    opt = torch.optim.Adam(
                        model.parameters(),
                        lr=model_args["lr"],
                        betas=(0.9, 0.95),
                        eps=1e-9,
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
                    model = load_jem((dev0, dev1), self.alphabet, model_args)
                else:
                    transformer_config = {
                        "heads": model_args["num_attns_heads"],
                        "head_dim": model_args["attn_head_dim"],
                        "dropout": model_args["dropout"],
                    }
                    model = BaselineModel(
                        dim=model_args["embedding_dim"],
                        alphabet=self.alphabet,
                        num_attn_layers=model_args["num_attn_layers"],
                        encoder_parallel_device=dev0,
                        decoder_parallel_device=dev1,
                        **transformer_config,
                    )
                print('Trainable Params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
            
            # > Optimizer
            if self.config.multimodal:
                if self.config.weight_decay:
                    opt = torch.optim.AdamW(
                        model.parameters(),
                        lr=model_args["lr"],
                        betas=(0.9, 0.95),
                        eps=1e-9,
                        weight_decay=model_args["weight_decay"],
                        )
                else:
                    opt = torch.optim.Adam(
                        model.parameters(),
                        lr=model_args["lr"],
                        betas=(0.9, 0.95),
                        eps=1e-9,
                    )

                if model_args["lr_scheduler"]:
                    lr_scheduler = WarmupLinearSchedule(optimizer=opt, **model_args["lr_scheduler"])
                else:
                    lr_scheduler = None
                    
            else:
                if self.config.weight_decay:
                    opt = torch.optim.AdamW(
                        model.parameters(),
                        lr=model_args["lr"],
                        betas=(0.9, 0.95),
                        eps=1e-9,
                        weight_decay=model_args["weight_decay"],
                        )
                else:
                    opt = torch.optim.Adam(
                        model.parameters(),
                        lr=model_args["lr"],
                        betas=(0.9, 0.95),
                        eps=1e-9,
                    )

                if model_args["lr_scheduler"]:
                    lr_scheduler = WarmupLinearSchedule(optimizer=opt, **model_args["lr_scheduler"])
                else:
                    lr_scheduler = None

        ############################
        #       Training Loop
        ############################
        # > Training loop variables
        if (self.distributed):
            if rank == 0:
                global_step = 0
                losses = [None, None]
                best_eval = float("inf")
                timestamp = datetime.now().strftime("%d-%m-%Y-%Hc%Mc%S")
        else:
            global_step = 0
            losses = [None, None]
            best_eval = float("inf")
            timestamp = datetime.now().strftime("%d-%m-%Y-%Hc%Mc%S")

        # > Initialize training and validation data loaders
        train_loader = self.training_loader(**dataloader_args)
        if (self.distributed):
            if rank == 0:
                val_loader = self.val_loader(**dataloader_args)
        else:
            val_loader = self.val_loader(**dataloader_args)

        for epoch_index in range(self.num_epochs):

            # > Initialize TQDM progress bar
            if (self.distributed):
                # ( ! ) Distributed DataLoader ( ONLY FOR DDP MULTI-GPU TRAINING )
                # > We need to repeat the random partition every epoch to guarantee randomness
                train_loader.sampler.set_epoch(epoch_index)

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
                if (self.distributed):
                    if rank == 0:
                        global_step += 1
                else:
                    global_step += 1

                # Skip Validation if unit test is enabled
                if _unit_test_enabled(self.unit_test):
                    # > Only update TQDM progress bar and log to wandb
                    if (self.distributed) and (self.config.multimodal):
                        if rank == 0:
                            # W&B logging
                            losses[0] = losses[0] / model_args["contrastive_loss_weight"]
                            losses[1] = losses[1] / model_args["cross_entropy_loss_weight"]
                            run.log({
                                "Learning Rate": opt.param_groups[0]['lr'],
                                "Total Loss": sum(losses),
                                "Contrastive Loss": losses[0],
                                "Cross-Entropy Loss": losses[1],
                            })
                            # TQDM logging
                            training_progress_bar.set_description(f"Rank {rank}: Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: (NA)")

                    elif (not self.distributed) and (self.config.multimodal):
                        # W&B logging
                        losses[0] = losses[0] / model_args["contrastive_loss_weight"]
                        losses[1] = losses[1] / model_args["cross_entropy_loss_weight"]
                        run.log({
                            "Learning Rate": opt.param_groups[0]['lr'],
                            "Total Loss": sum(losses),
                            "Contrastive Loss": losses[0],
                            "Cross-Entropy Loss": losses[1],
                        })
                        # TQDM logging
                        training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: (NA)")

                    elif (self.distributed) and (not self.config.multimodal):
                        if rank == 0:
                            # W&B logging
                            run.log({
                                "Learning Rate": opt.param_groups[0]['lr'],
                                "Cross-Entropy Loss": losses[0],
                            })
                            # TQDM logging
                            training_progress_bar.set_description(f"Rank {rank}: Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: (NA)")

                    else:
                        # W&B logging
                        run.log({
                            "Learning Rate": opt.param_groups[0]['lr'],
                            "Cross-Entropy Loss": losses[0],
                        })
                        # TQDM logging
                        training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: (NA)")

                else:
                    # > Logging training and validation loss
                    if (self.distributed) and (self.config.multimodal):
                        if rank == 0:
                            if (global_step % self.config.log_interval) == 0:
                                # Evaluate on validation set
                                if (global_step % self.config.val_interval) == 0:
                                    val_loss = self.evaluate(val_loader, model)
                                    run.log({"Validation Set Loss": val_loss})

                                    if best_eval > val_loss:
                                        best_eval = val_loss
                                        checkpoint_state = {
                                                'epoch': epoch_index,
                                                'model_state_dict': model.state_dict(),
                                                'optimizer_state_dict': opt.state_dict(),
                                                'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                                                }
                                        torch.save(
                                                checkpoint_state,
                                                os.path.join(run_dir, f'{self.config.name}_ckpt_{timestamp}.pth')
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
                                })

                    elif (not self.distributed) and (self.config.multimodal):
                        if (global_step % self.config.log_interval) == 0:
                            # Evaluate on validation set
                            if (global_step % self.config.val_interval) == 0:
                                val_loss = self.evaluate(val_loader, model)
                                run.log({"Validation Set Loss": val_loss})

                                if best_eval > val_loss:
                                    best_eval = val_loss
                                    checkpoint_state = {
                                            'epoch': epoch_index,
                                            'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': opt.state_dict(),
                                            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                                            }
                                    torch.save(
                                            checkpoint_state,
                                            os.path.join(run_dir, f'{self.config.name}_ckpt_{timestamp}.pth')
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
                            })

                    elif (self.distributed) and (not self.config.multimodal):
                        if rank == 0:
                            if (global_step % self.config.log_interval) == 0:
                                # Evaluate on validation set
                                if (global_step % self.config.val_interval) == 0:
                                    val_loss = self.evaluate(val_loader, model)
                                    run.log({"Validation Set Loss": val_loss})

                                    if best_eval > val_loss:
                                        best_eval = val_loss
                                        checkpoint_state = {
                                                'epoch': epoch_index,
                                                'model_state_dict': model.state_dict(),
                                                'optimizer_state_dict': opt.state_dict(),
                                                'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                                                }
                                        torch.save(
                                                checkpoint_state,
                                                os.path.join(run_dir, f'{self.config.name}_ckpt_{timestamp}.pth')
                                                )

                                # Update TQDM progress bar
                                training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: ({best_eval})")

                                # Log to wandb
                                run.log({
                                    "Learning Rate": opt.param_groups[0]['lr'],
                                    "Cross-Entropy Loss": losses[0],
                                })

                    else:
                        if (global_step % self.config.log_interval) == 0:
                            # Evaluate on validation set
                            if (global_step % self.config.val_interval) == 0:
                                val_loss = self.evaluate(val_loader, model)
                                run.log({"Validation Set Loss": val_loss})

                                if best_eval > val_loss:
                                    best_eval = val_loss
                                    checkpoint_state = {
                                            'epoch': epoch_index,
                                            'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': opt.state_dict(),
                                            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                                            }
                                    torch.save(
                                            checkpoint_state,
                                            os.path.join(run_dir, f'{self.config.name}_ckpt_{timestamp}.pth')
                                            )
                            
                            # Update TQDM progress bar
                            training_progress_bar.set_description(f"Epoch {epoch_index + 1}, Loss ({sum(losses)}), Best Val Loss: ({best_eval})")

                            # Log to wandb
                            run.log({
                                "Learning Rate": opt.param_groups[0]['lr'],
                                "Cross-Entropy Loss": losses[0],
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
