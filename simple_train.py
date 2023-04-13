import os
import json
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import wandb
import random
import numpy as np
from tqdm import tqdm

from pjm.model.training_utils import load_model
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
    parser.add_argument("--log_interval", type=int, default=100, help="Set logging frequency.")
    parser.add_argument("--tags", nargs="+", help="W&B run tags.")
    parser.add_argument("--detect_anomaly", action="store_true", help="Set to detect anomaly.")
    return parser.parse_args()


def main():
    args = parse()

    seed_everything(2121992)

    dataset = AF2SCN(
	"train",
	max_len=args.max_len,
	dataset_path=args.dataset_path,
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
	dataset,
	batch_size=args.batch_size,
	num_workers=args.num_workers,
	collate_fn=collate_fn,
	shuffle=True,
    )

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

    nan_watcher = 0
    prev_batch_pdb_ids = None
    with wandb.init(dir=".", project="jessy", tags=args.tags):
        #wandb.watch(model, log_freq=args.log_interval)
        
        for epoch in range(args.num_epochs):
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}: Loss (NA)",
                mininterval=args.log_interval
            )

            for batch_index, batch in progress_bar:
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
                    ct_loss, ce_loss = model(
                        sequences,
                        structures,
                    )
                    total = ct_loss + ce_loss

                    total.backward()

                opt.step()
                
                ce_loss = ce_loss.detach().item()
                ct_loss = ct_loss.detach().item()
                wandb.log({
                    "Masked Cross-Entropy": ce_loss,
                    "Contrastive": ct_loss,
                    "Total": total.item()
                })

                if (batch_index + 1) % args.log_interval == 0:
                    progress_bar.set_description(f"Epoch {epoch + 1}: Loss ({ce_loss + ct_loss})")

                    checkpoint_state = {
                            'model_state_dict': model.state_dict(),
                    }
                    torch.save(
                            checkpoint_state,
                            os.path.join(args.model_chkpt_path, f'model_chkpt_epoch{epoch + 1}.pth')
                    )


if __name__ == "__main__":
    main()
