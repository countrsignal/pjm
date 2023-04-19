import os
import json
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import wandb
import random
import numpy as np
from tqdm import tqdm

from pjm.model import BaselineModel
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
    parser.add_argument("-l", "--learning_rate", type=float, default=3e-4, help="Set learning rate.")
    parser.add_argument("-e", "--num_epochs", type=int, default=100, help="Set number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Set batch size.")
    parser.add_argument("-m", "--max_len", type=int, default=1022, help="Set max size of proteins.")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="Set number of workers fo dataloaders")
    parser.add_argument("-p", "--prefetch_factor", default=100000, type=int, help="Determine how many batches to cache.")
    parser.add_argument("--num_layers", default=8, type=int, help="Number of attention heads")
    parser.add_argument("--num_heads", default=8, type=int, help="Number of attention heads")
    parser.add_argument("--dim_head", default=128, type=int, help="Dimension for each attention head")
    parser.add_argument("--load_from_chkpt", type=str, help="Load from checkpoint.")
    parser.add_argument("--log_interval", type=int, default=100, help="Set logging frequency.")
    parser.add_argument("--tags", nargs="+", help="W&B run tags.")
    parser.add_argument("--detect_anomaly", action="store_true", help="Set to detect anomaly.")
    parser.add_argument("--overfit", action="store_true", help="Activate overfitting unit test.")
    parser.add_argument("--adamw", action="store_true", help="Use AdamW as opposed to regular Adam.")
    return parser.parse_args()


def main():
    args = parse()

    seed_everything(2121992)

    dataset = AF2SCN(
        "train",
        max_len=args.max_len,
        dataset_path=args.dataset_path,
        _sequence_only_baseline=True,
        _filter_by_plddt_coverage=None,
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
    if args.overfit:
        batch = next(iter(train_loader))
        train_loader = [batch]
        args.log_interval = 1

    transformer_config = {
        "depth": 1,
        "heads": args.num_heads,
        "dim_head": args.dim_head,
        "dropout": 0.1,
    }
    model = BaselineModel(
        args.dim_head,
        alphabet,
        args.num_layers,
        "cuda:0",
        "cuda:1",
        **transformer_config,
    )

    if args.load_from_chkpt:
        model.load_state_dict(torch.load(args.load_from_chkpt)["model_state_dict"])

    model.dispatch_params()

    if args.adamw:
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01, fused=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    with wandb.init(dir=".", project="baseline", tags=args.tags):
        
        for epoch in range(args.num_epochs):
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}: Loss (NA)",
                mininterval=args.log_interval
            )

            for batch_index, batch in progress_bar:
                if (args.overfit):
                    if (epoch == 0):
                        # Overfitting unit test
                        sequences = batch.seqs.to(0)
                else:
                    sequences = batch.seqs.to(0)

                opt.zero_grad()

                # Hunt for NaN's in the backward pass
                if args.detect_anomaly:
                    with torch.autograd.detect_anomaly():
                        ce_loss = model(sequences)
                        ce_loss.backward()
                else:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                        ce_loss = model(sequences)
                    
                    ce_loss.backward()

                opt.step()
                
                ce_loss = ce_loss.detach().item()
                wandb.log({
                    "Masked Cross-Entropy": ce_loss,
                })

                if (batch_index + 1) % args.log_interval == 0:
                    progress_bar.set_description(f"Epoch {epoch + 1}: Loss ({ce_loss})")

                    # checkpoint_state = {
                    #         'model_state_dict': model.state_dict(),
                    # }
                    # torch.save(
                    #         checkpoint_state,
                    #         os.path.join(args.model_chkpt_path, f'model_chkpt_epoch{epoch + 1}.pth')
                    # )


if __name__ == "__main__":
    main()
