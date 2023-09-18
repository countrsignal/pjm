import os
import json
import logging
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import wandb
import random
import numpy as np
from tqdm import tqdm

from pjm.utils.training_utils import _setup_logger, load_jem
from pjm.utils import Collator, Alphabet, AF2SCN


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


def _check_nan_sanity(loss):
    return (torch.isnan(loss) or torch.isinf(loss))


def validate(args, bb_only, model, val_loader):
    # Averages
    total_list = []
    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            if not args.sanity_check:
                sequences, *structures = batch.process_data(0)
                if args.structure_test:
                    g, n, e = structures
                    ns, nv = n
                    nv = nv.mean(dim=-2, keepdim=True)
                    ns = ns.index_select(dim=-1, index=bb_only)
                    n = (ns, nv)
                    structures = (g, n, e)
            else:
                sequences = batch.seqs.to("cuda:0")
                structures = None

            if (args.bf16) and (torch.cuda.is_bf16_supported()):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    loss_dict = model(
                        sequences,
                        structures,
                    )
            else:
                loss_dict = model(
                    sequences,
                    structures,
                )
            
            total = loss_dict["total"]
            if _check_nan_sanity(total):
                logging.warning(f"Validation loss is NaN. Skipping...")
                continue
            
            total = total.detach().item()
            total_list.append(total)
    
    # Log validation metrics
    # > Calculate averages for logging
    avg_total = sum(total_list) / len(total_list)
    # > Log to W&B
    wandb.log({
        "Validation Set Loss": avg_total,
    })
    
    # Return model to training mode
    model.train()

    return avg_total


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



def parse():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Set path to pre-computed data.")
    parser.add_argument("--model_chkpt_path", type=str, help="Set path to model previously trained model.")
    parser.add_argument("--model_config_path", type=str, help="Set path to model config json.")
    parser.add_argument("-e", "--num_epochs", type=int, default=100, help="Set number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Set batch size.")
    parser.add_argument("-m", "--max_len", type=int, default=1022, help="Set max size of proteins.")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="Set number of workers fo dataloaders")
    parser.add_argument("-p", "--prefetch_factor", default=100000, type=int, help="Determine how many batches to cache.")
    parser.add_argument("--load_from_chkpt", type=str, help="Load from checkpoint.")
    parser.add_argument("--log_interval", type=int, default=50000, help="Set logging frequency.")
    parser.add_argument("--val_interval", type=int, default=50000, help="Set logging frequency.")
    parser.add_argument("--tags", nargs="+", help="W&B run tags.")
    parser.add_argument("--detect_anomaly", action="store_true", help="Set to detect anomaly.")
    parser.add_argument("--bf16", action="store_true", help="Enable mixed precision training (for BFloat16 ONLY).")
    parser.add_argument("--fwd_test", action="store_true", help="Run a simple forward pass unit test.")
    parser.add_argument("--structure_test", action="store_true", help="Test whether the model is cheating counting padding dimensions.")
    parser.add_argument("--sanity_check", action="store_true", help="Test the Joint Embeddings model code by training it as a PLM.")
    parser.add_argument("--run_name", type=str, help="Name for the experiment.")
    return parser.parse_args()


def main():
    args = parse()

    seed_everything(2121992)

    train_ds = AF2SCN(
	"train",
	max_len=args.max_len,
	dataset_path=args.dataset_path,
    _sequence_only_baseline=args.sanity_check,
	_filter_by_plddt_coverage=None
    )
    val_ds = AF2SCN(
	"val",
	max_len=args.max_len,
	dataset_path=args.dataset_path,
    _sequence_only_baseline=args.sanity_check,
	_filter_by_plddt_coverage=None
    )

    alphabet = Alphabet(
	prepend_toks = ("<sos>", "<eos>", "<unk>"),
	append_toks = ("<cls>", "<pad>", "<mask>"),
	prepend_bos = True,
	append_eos = True,
	use_msa = False
    )
    collate_fn = Collator(tokenizer=alphabet.get_batch_converter())
    train_loader = DataLoader(
	train_ds,
	batch_size=args.batch_size,
	num_workers=args.num_workers,
	collate_fn=collate_fn,
	shuffle=True,
    )
    val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    shuffle=False,
    )

    # Create a single batch for forward pass unit test
    if (args.fwd_test):
        batch = next(iter(train_loader))

    model_args = json.load(open(args.model_config_path, "r"))
    if not args.sanity_check:
        if args.structure_test:
            print("\nBackbone only model is selected!\n")
            bb_only = [0, 1, 2, 12, 13, 14]
            # for i in range(24, 77):
            #     bb_only.append(i)
            bb_only = torch.LongTensor(bb_only)
            bb_only = bb_only.to("cuda:0")
            model_args["node_in_dims"][0] = bb_only.shape[0]
            model_args["node_in_dims"][1]  = 1
        else:
            bb_only = None
    else:
        bb_only = None

    model = load_jem(
	devices=(0,1),
	alphabet=alphabet,
	model_args=model_args,
    _sanity_check_mode_=args.sanity_check,
    )

    if args.load_from_chkpt:
        # most_recent_chkpt = sorted(os.listdir(args.model_chkpt_path), reverse=True)[0]
        # model.load_state_dict(torch.load(most_recent_chkpt["model_state_dict"]))
        model.load_state_dict(torch.load(args.load_from_chkpt)["model_state_dict"])

    model.dispatch_params()

    if model_args["weight_decay"] == 0.0:
        opt = torch.optim.Adam(model.parameters(), lr=model_args["lr"], betas=(0.9, 0.98), eps=1e-9)
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=model_args["lr"],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=model_args["weight_decay"]
        )

    if model_args["lr_scheduler"]:
        lr_scheduler = WarmupLinearSchedule(optimizer=opt, **model_args["lr_scheduler"])
    else:
        lr_scheduler = None

    # nan_watcher = 0
    # prev_batch_pdb_ids = None
    log_step = 0
    best_val_loss = float("inf")
    total_adjusted = float("inf")
    with wandb.init(dir=".", project="joint embeddings", name=args.run_name, tags=args.tags, config=model_args):
        
        for epoch in range(args.num_epochs):
            if (args.fwd_test):
                progress_bar = tqdm(
                    enumerate([batch]),
                    total=1,
                    desc=f"Epoch {epoch + 1}: Train Loss ({total_adjusted}), Best Val Loss ({best_val_loss})",
                    mininterval=args.log_interval
                )
            else:                
                progress_bar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}: Train Loss ({total_adjusted}), Best Val Loss ({best_val_loss})",
                    mininterval=args.log_interval
                )

            for batch_index, batch in progress_bar:
                if (args.fwd_test):
                    if (epoch == 1):
                        break

                if not args.sanity_check:
                    sequences, *structures = batch.process_data(0)
                    if args.structure_test:
                        g, n, e = structures
                        ns, nv = n
                        nv = nv.mean(dim=-2, keepdim=True)
                        ns = ns.index_select(dim=-1, index=bb_only)
                        n = (ns, nv)
                        structures = (g, n, e)
                else:
                    sequences = batch.seqs.to("cuda:0")
                    structures = None

                opt.zero_grad()

                # Hunt for NaN's in the backward pass
                if args.detect_anomaly:
                    with torch.autograd.detect_anomaly():
                        loss_dict = model(
                            sequences,
                            structures,
                        )
                        total = loss_dict["total"]
                        total.backward()
                        
#                        total = 0.
#                        for loss_name, loss in [("Contrastive", ct_loss), ("Autoregressive", ce_loss)]:
#                            if _check_nan_sanity(loss):
#                                trigger = True
#                                logging.warning(f"{loss_name} loss is NaN. Skipping...")
#                                loss = loss.new_tensor(0., requires_grad=True)
#                            else:
#                                trigger = False
#                            total = total + loss

 #                       if (nan_watcher < 1) and (trigger):
 #                           nan_watcher += 1
 #                           prev_ = '\n\t'.join(prev_batch_pdb_ids)
 #                           curr_ = '\n\t'.join(batch.pids)
 #                           print(f"NaN encounter at batch {batch_index} of epoch {epoch}\n")
 #                           print(f"Logging PDB ID's of the PREVIOUS batch:\n{prev_}")
 #                           print(f"Logging PDB ID's of the CURRENT batch:\n{curr_}")
 #                           wandb.log({
 #                               "batch_index": batch_index,
 #                               "prev_batch": wandb.Table(prev_.split("\n\t")),
 #                               "curr_batch": wandb.Table(curr_.split("\n\t"))
 #                           })
 #                       else:
 #                           prev_batch_pdb_ids = batch.pids
                else:
                    if (args.bf16) and (torch.cuda.is_bf16_supported()):
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                            loss_dict = model(
                                sequences,
                                structures,
                            )
                    else:
                        loss_dict = model(
                            sequences,
                            structures,
                        )
                    total = loss_dict["total"]
                    total.backward()

                opt.step()
                if lr_scheduler:
                    lr_scheduler.step()

                log_step += 1
                if log_step % args.log_interval == 0:
                    ce_loss = loss_dict["cross entropy"]
                    ce_loss = ce_loss.detach().item()
                    ce_loss = ce_loss / model_args["cross_entropy_loss_weight"]
                    
                    if not args.sanity_check:
                        ct_loss = loss_dict["contrastive"]
                        ct_loss = ct_loss.detach().item()
                        ct_loss = ct_loss / model_args["contrastive_loss_weight"]
                        total_adjusted = ce_loss + ct_loss
                        wandb.log({
                            "Cross-Entropy Loss": ce_loss,
                            "Contrastive Loss": ct_loss,
                            "Total Loss": total_adjusted,
                            "Learning Rate": opt.param_groups[0]["lr"],
                        })
                    else:
                        total_adjusted = ce_loss
                        wandb.log({
                            "Cross-Entropy Loss": ce_loss,
                            "Learning Rate": opt.param_groups[0]["lr"],
                        })
                    
                    if log_step % args.val_interval == 0:
                        avg_val_loss = validate(args, bb_only, model, val_loader)

                        if best_val_loss > avg_val_loss:
                            best_val_loss = avg_val_loss
                            checkpoint_state = {
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': opt.state_dict(),
                                    'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                            }
                            model_name = args.run_name.split("_")
                            model_name.insert(-1, f"{epoch}epochs")
                            model_name = "_".join(model_name)
                            torch.save(
                                    checkpoint_state,
                                    os.path.join(args.model_chkpt_path, f"{model_name}_ckpt.pth")
                            )
                    
                    progress_bar.set_description(f"Epoch {epoch + 1}: Train Loss ({total_adjusted}), Best Val Loss ({best_val_loss})")


if __name__ == "__main__":
    main()
