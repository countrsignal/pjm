import os
import json
from argparse import ArgumentParser


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import torch

from pjm.utils import launch_ddp, Overfit, EarlyStopping, Pipeline


def parse():
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int,
                        default=64, help="Set batch size.")
    parser.add_argument("-e", "--num_epochs", type=int,
                        help="Set number of training epochs.")
    parser.add_argument("-m", "--max_len", type=int,
                        default=1022, help="Set max size of proteins.")
    parser.add_argument("-w", "--num_workers", type=int,
                        default=8, help="Set number of workers fo dataloaders")
    parser.add_argument("-p", "--prefetch_factor", type=int,
                        help="Determine how many batches to cache.")
    parser.add_argument("-o", "--overfit",
                        action="store_true", help="Overfit unit test.")
    parser.add_argument("-s", "--early_stopping",
                        action="store_true", help="Overfit uni test.")
    parser.add_argument("--dataset_path", type=str, help="Set path to pre-computed data.")
    parser.add_argument("--plddt_filter", type=float,
                        default=0.20, help="Filter proteins by pLDDT coverage.")
    parser.add_argument("--model_config_path", type=str, help="Set path to model config json.")
    parser.add_argument("--log_interval", type=int,
                        default=5, help="Set logging frequency.")
    parser.add_argument("--tags", nargs="+", help="W&B run tags.")
    parser.add_argument("--seed", type=int, help="Set random seed.")
    return parser.parse_args()


def main(num_gpus, enable_parallel_training):
    # Parse arguements
    args = parse()

    ############
    #   Setup
    ############
    # > DataLoader args
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_args.update({"prefetch_factor": args.prefetch_factor})

    ############
    #   Train
    ############
    model_args = json.load(open(args.model_config_path))
    pipe = Pipeline(
        training_args=args,
        parallel_training=enable_parallel_training,
    )
    if not enable_parallel_training:
        pipe.fit(
            rank=None,
            world_size=None,
            model_args=model_args,
            dataloader_args=loader_args,
            unit_test_callback=Overfit() if args.overfit else None,
            early_stop_callback=EarlyStopping() if args.early_stopping else None,
        )
    else:
        launch_ddp(
            training_fn=pipe.fit,
            world_size= int(num_gpus / 2),
            model_args=model_args,
            dataloader_args=loader_args,
            unit_test_callback=Overfit() if args.overfit else None,
            early_stop_callback=EarlyStopping() if args.early_stopping else None,
        )


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    print(f"\nFound {n_gpus} GPUs available.\n")
    if n_gpus > 2:
        assert (n_gpus % 2) == 0, "Number of GPUs must be even."
        main(n_gpus, enable_parallel_training=True)
    else:
        main(n_gpus, enable_parallel_training=False)
