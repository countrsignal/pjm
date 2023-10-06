import os
import logging
from functools import partial

import wandb
from torch.cuda import is_bf16_supported, device_count

from pjm.utils.training_utils import (
    set_training_env_vars,
    seed_everything,
    _setup_logger,
    training_parser,
    build_optimizer,
    assembler,
    training_step_hook,
    logging_hook,
    init_progress_bar,
    init_runner,
    write_to_logger,
    EvalMonitor,
)


def main():
    args = training_parser()
    seed_everything(args.seed)
    info_log = _setup_logger(logger_name="GENERAL", log_level=logging.INFO)
    info_log.info(f"Number of GPUs: {device_count()}")
    info_log.info(f"BFloat16 supported: {is_bf16_supported()}")

    model_type = "mmplm" if args.multi_modal else "baseline"
    info_log.info(f"Initializing {model_type.upper()} model based on {args.config_path}")
    config, train_loader, model = assembler(args.config_path, args.multi_modal)
    info_log.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.register_devices(*args.gpu_devices)
    optimizer, lr_scheduler = build_optimizer(model, config)
    model.dispatch_params()

    run = init_runner(args.run_name, args.tags, config["model"][model_type])
    val_monitor = EvalMonitor(
        split="validation",
        multi_modal=args.multi_modal,
        monitor_interval=args.val_interval,
        config=config["model"][model_type]
    )

    # Training loop
    global_step = 0
    prev_train_loss = float("inf")
    if args.multi_modal:
        # NOTE: for multi-modal PLM, sequences AND structures are sent to decoder module first
        input_device = model.decoder_parallel_device
    else:
        # NOTE: for baseline PLM, sequences are sent to encoder module first
        input_device = model.encoder_parallel_device
    info_log.info("Beginning training.")
    for epoch_index in range(args.epochs):
        # TQDM progress bar
        progress_bar = init_progress_bar(
            epoch_index,
            prev_train_loss,
            val_monitor.best_eval_loss,
            args.log_interval,
            train_loader,
        )
        for batch_index, batch in progress_bar:
            # Training
            sequences, *structures = batch.process_data(input_device, multi_modal=args.multi_modal)
            train_step = partial(training_step_hook, args.debug, args.multi_modal, model, optimizer, lr_scheduler)
            losses = train_step(sequences, structures) if args.multi_modal else train_step(sequences)
            global_step += 1

            # Logging
            if (global_step % args.log_interval) == 0:
                log_dict = logging_hook(args.multi_modal, losses, lr_scheduler)
                write_to_logger(run, log_dict)
                # Update progress bar
                prev_train_loss = log_dict["Training Loss"] if args.multi_modal else log_dict["Masked Residue Loss"]
                progress_bar.set_description(f"Epoch {epoch_index + 1} | Training Step Loss {prev_train_loss:.4f} | Best Validation Loss {val_monitor.best_eval_loss:.4f}")

                # Validation
                if not args.debug:
                    if val_monitor.watch(global_step):
                        val_monitor.evaluation_step(epoch_index, run, model, optimizer, lr_scheduler, ckpt_model=True)

    info_log.info("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    set_training_env_vars()
    main()