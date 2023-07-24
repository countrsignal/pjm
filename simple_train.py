import os
import json
import logging
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import wandb
import random
import numpy as np
from tqdm import tqdm

from pjm.model.training_utils import _setup_logger, load_model
from pjm.utils import Collator, Alphabet, AF2SCN


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _check_sanity(loss):
    return (torch.isnan(loss) or torch.isinf(loss))


def validate(model, val_loader):
    # Averages
    ce_loss = []
    ct_loss = []
    total = []
    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            sequences, *structures = batch.process_data(0)
            ct_loss, ce_loss = model(
                sequences,
                structures,
            )
            total = ct_loss + ce_loss
            if _check_sanity(total):
                logging.warning(f"Validation loss is NaN. Skipping...")
                continue
            
            ce_loss = ce_loss.detach().item()
            ct_loss = ct_loss.detach().item()
            total = total.detach().item()
            ce_loss.append(ce_loss)
            ct_loss.append(ct_loss)
            total.append(total)
    
    # Log validation metrics
    # > Calculate averages for logging
    ce_loss = sum(ce_loss) / len(ce_loss)
    ct_loss = sum(ct_loss) / len(ct_loss)
    total = sum(total) / len(total)
    # > Log to W&B
    wandb.log({
        "Validation Masked Cross-Entropy": ce_loss,
        "Validation Contrastive": ct_loss,
        "Validation Total": total,
    })
    
    # Return model to training mode
    model.train()

    return total



def parse():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Set path to pre-computed data.")
    parser.add_argument("--model_chkpt_path", type=str, help="Set path to model previously trained model.")
    parser.add_argument("--model_config_path", type=str, help="Set path to model config json.")
    parser.add_argument("-l", "--learning_rate", type=float, default=3e-4, help="Set learning rate.")
    parser.add_argument("-e", "--num_epochs", type=int, default=100, help="Set number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Set batch size.")
    parser.add_argument("-m", "--max_len", type=int, default=1022, help="Set max size of proteins.")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="Set number of workers fo dataloaders")
    parser.add_argument("-p", "--prefetch_factor", default=100000, type=int, help="Determine how many batches to cache.")
    parser.add_argument("--load_from_chkpt", type=str, help="Load from checkpoint.")
    parser.add_argument("--log_interval", type=int, default=50000, help="Set logging frequency.")
    parser.add_argument("--tags", nargs="+", help="W&B run tags.")
    parser.add_argument("--detect_anomaly", action="store_true", help="Set to detect anomaly.")
    parser.add_argument("--bf16", action="store_true", help="Enable mixed precision training (for BFloat16 ONLY).")
    parser.add_argument("--fwd_test", action="store_true", help="Run a simple forward pass unit test.")
    return parser.parse_args()


def main():
    args = parse()

    seed_everything(2121992)

    train_ds = AF2SCN(
	"train",
	max_len=args.max_len,
	dataset_path=args.dataset_path,
	_filter_by_plddt_coverage=None
    )
    # val_ds = AF2SCN(
	# "val",
	# max_len=args.max_len,
	# dataset_path=args.dataset_path,
	# _filter_by_plddt_coverage=None
    # )

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
    # val_loader = DataLoader(
    # val_ds,
    # batch_size=args.batch_size,
    # num_workers=args.num_workers,
    # collate_fn=collate_fn,
    # shuffle=False,
    # )

    # Create a single batch for forward pass unit test
    if (args.fwd_test):
        batch = next(iter(train_loader))

    model_args = json.load(open(args.model_config_path, "r"))
    model = load_model(
	devices=(0,1),
	alphabet=alphabet,
	model_args=model_args
    )

    if args.load_from_chkpt:
        # most_recent_chkpt = sorted(os.listdir(args.model_chkpt_path), reverse=True)[0]
        # model.load_state_dict(torch.load(most_recent_chkpt["model_state_dict"]))
        model.load_state_dict(torch.load(args.load_from_chkpt)["model_state_dict"])

    model.dispatch_params()

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    # opt = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     betas=(0.9, 0.98),
    #     eps=1e-9,
    #     weight_decay=model_args["weight_decay"]
    # )

    # nan_watcher = 0
    # prev_batch_pdb_ids = None
    log_step = 0
    best_val_loss = float("inf")
    total = torch.Tensor([float("inf")])
    with wandb.init(dir=".", project="joint embeddings", name="simple train", tags=args.tags):
        #wandb.watch(model, log_freq=args.log_interval)
        
        for epoch in range(args.num_epochs):
            if (args.fwd_test):
                progress_bar = tqdm(
                    enumerate([batch]),
                    total=1,
                    desc=f"Epoch {epoch + 1}: Loss ({total.item()})",
                    mininterval=args.log_interval
                )
            else:                
                progress_bar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}: Loss ({total.item()})",
                    mininterval=args.log_interval
                )

            for batch_index, batch in progress_bar:
                if (args.fwd_test):
                    if (epoch == 1):
                        break

                sequences, *structures = batch.process_data(0)

                opt.zero_grad()

                # Hunt for NaN's in the backward pass
                if args.detect_anomaly:
                    with torch.autograd.detect_anomaly():
                        ct_loss, ce_loss = model(
                            sequences,
                            structures,
                        )
                        total = ct_loss + ce_loss
                        total.backward()
                        
#                        total = 0.
#                        for loss_name, loss in [("Contrastive", ct_loss), ("Autoregressive", ce_loss)]:
#                            if _check_sanity(loss):
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
                    if (args.bf16):
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                            ct_loss, ce_loss, total = model(
                                sequences,
                                structures,
                            )
                    else:
                        ct_loss, ce_loss, total = model(
                            sequences,
                            structures,
                        )

                    total.backward()

                opt.step()

                log_step += 1
                if log_step % args.log_interval == 0:
                    ce_loss = ce_loss.detach().item()
                    ct_loss = ct_loss.detach().item()
                    wandb.log({
                        "Cross-Entropy Loss": ce_loss,
                        "Contrastive Loss": ct_loss,
                        "Total Loss": total.item()
                    })
                    # with torch.no_grad():
                    #     if (args.bf16):
                    #         with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    #             seq_embs, strc_embs = model(
                    #                 sequences,
                    #                 structures,
                    #                 return_embeddings=True,
                    #             )
                    #     else:
                    #         seq_embs, strc_embs = model(
                    #             sequences,
                    #             structures,
                    #             return_embeddings=True,
                    #         )
                    
                    # avg_val_loss = validate(model, val_loader)
                    # progress_bar.set_description(f"Epoch {epoch + 1}: Train Loss ({total.item()}), Val Loss ({avg_val_loss})")

                    # if best_val_loss > avg_val_loss:
                    #     best_val_loss = avg_val_loss
                    #     checkpoint_state = {
                    #             'model_state_dict': model.state_dict(),
                    #     }
                    #     torch.save(
                    #             checkpoint_state,
                    #             os.path.join(args.model_chkpt_path, 'model_chkpt_simple_train.pth')
                    #     )


if __name__ == "__main__":
    main()
