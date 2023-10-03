import os
from functools import partial

import wandb

from pjm.utils.training_utils import (
    set_training_env_vars,
    seed_everything,
    training_parser,
    build_optimizer,
    assembler,
    training_step_hook,
    logging_hook,
    validation_step_hook,
    init_progress_bar,
    init_runner,
    write_to_logger,
    write_checkpoint,
)


def main():
    args = training_parser()
    seed_everything(args.seed)

    config, train_loader, val_loader, model = assembler(args.config_path, args.multi_modal)
    model.register_devices(*args.gpu_devices)
    optimizer, lr_scheduler = build_optimizer(model, config)
    model.dispatch_params()

    model_type = "mmplm" if args.multi_modal else "baseline"
    run = init_runner(args.run_name, args.tags, config["model"][model_type])

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    prev_train_loss = float("inf")
    if args.multi_modal:
        # NOTE: for multi-modal PLM, sequences AND structures are sent to decoder module first
        input_device = model.decoder_parallel_device
    else:
        # NOTE: for baseline PLM, sequences are sent to encoder module first
        input_device = model.encoder_parallel_device
    for epoch_index in range(args.epochs):
        # TQDM progress bar
        progress_bar = init_progress_bar(
            epoch_index,
            prev_train_loss,
            best_val_loss,
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
                progress_bar.set_description(f"Epoch {epoch_index + 1} | Training Step Loss {prev_train_loss:.4f} | Best Validation Loss {best_val_loss:.4f}")

                # Validation
                if not args.debug:
                    if (global_step % args.val_interval) == 0:
                        val_log_dict = validation_step_hook(args.multi_modal, val_loader, model)
                        run.log(val_log_dict)
                        val_loss = val_log_dict["Validation Loss"]
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            write_checkpoint(
                                model,
                                optimizer,
                                lr_scheduler,
                                epoch_index,
                                config,
                                os.path.join(run.dir, f"{args.run_name}_ckpt.pth"),
                            )
    wandb.finish()


if __name__ == "__main__":
    set_training_env_vars()
    main()