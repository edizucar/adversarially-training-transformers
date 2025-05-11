#!/usr/bin/env python

import os
import torch
import wandb
import signal
import argparse

from src.model import GPTConfig, GPT
from src.config import TrainingConfig
from src.probes import ProbeCluster
import src.utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('--config', type=str, help='Path to config file (YAML or Python)')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--lambda_adversarial', type=float, help='Lambda weighting of probe loss for adversarial training')
    parser.add_argument('--phi_probe_steps_per_model_update', type=float, help='Phi probe steps per model update')
    parser.add_argument('--max_iters', type=int, help='Maximum number of iterations')
    return parser.parse_args()

def main():
    # Basic setup
    args = parse_args()
    if args.config.endswith('.yaml'):
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig.from_py(args.config)

    if args.no_wandb:
        config.wandb_log = False
    if args.eval_only:
        config.eval_only = True
    if args.lambda_adversarial is not None:
        config.lambda_adversarial = args.lambda_adversarial
    if args.phi_probe_steps_per_model_update is not None:
        config.phi_probe_steps_per_model_update = args.phi_probe_steps_per_model_update
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    config.eval_interval = min(config.eval_interval, config.max_iters)
    config.log_interval = min(config.log_interval, config.max_iters)
    config.lr_decay_iters = min(config.lr_decay_iters, config.max_iters)
    
    # Initialize timing tracker
    timer = utils.TimingTracker()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    config.device = device
    
    config, ctx, ptdtype = utils.setup_environment(config, device_type)
    
    # Load dataset
    data_dir = os.path.join('data', config.dataset)
    train_data, val_data = utils.load_dataset(data_dir)
    
    # Get model args
    config, checkpoint = utils.get_model_args(config, data_dir, device)
    
    if config.init_from == 'huggingface':
        block_size = config.hf_config_block_size
    else:
        block_size = config.block_size

    iter_num = 0
    best_val_loss = float('inf')

    # Load model from checkpoint if needed
    if config.init_from == 'huggingface':
        model, model_args = utils.load_model_from_huggingface(config.huggingface_model_id, device, GPT, GPTConfig)
    elif config.init_from == 'resume' and checkpoint is not None:
        model, model_args, _, iter_num, best_val_loss = utils.load_model_from_checkpoint(checkpoint, device, GPT, GPTConfig)
    elif config.init_from == 'resume' and checkpoint is None:
        raise ValueError(f"Checkpoint not found for resume")
    else:
        raise ValueError(f"Invalid init_from value: {config.init_from}")
    
    # Intialize probes if needed
    probe_cluster = utils.initialize_probes(config, device, checkpoint, ProbeCluster) if config.train_probes else None
    
    # Setup training components
    model, optimizer, scaler = utils.setup_training_components(model, config, device_type)
    
    # Create batch getter
    get_batch_bound = utils.create_batch_getter(train_data, val_data, config, device, timer, utils.get_batch)
    
    # Tune batch size if requested
    if config.auto_tune_batch_size:
        config.batch_size = utils.auto_tune_batch_size(
            model, config, ctx, optimizer, get_batch_bound, probe_cluster
        )

    if config.wandb_log:
        utils.setup_wandb(config)

    stop_requested, signal_handler = utils.setup_signal_handler()
    signal.signal(signal.SIGINT, signal_handler)

    # Main training loop
    print("\nStarting training...")
    
    # Track total iteration time
    timer.start('total_training')
    
    while iter_num < config.max_iters:
        timer.start('iteration')
        
        # Learning rate decay
        timer.start('lr_update')
        lr = utils.update_learning_rate(optimizer, config, iter_num)
        timer.end('lr_update')

        if iter_num % config.eval_interval == 0 and (iter_num > 0 or 'PROFILE_RUN' not in os.environ):
            eval_loss, probe_losses = utils.run_evaluation(model, probe_cluster, config, ctx, get_batch_bound, timer)

            utils.log_evaluation_results(eval_loss, probe_losses, iter_num, lr, config)

            # Save checkpoint if needed
            if iter_num > 0 and (not config.never_save_checkpoint and eval_loss['val'] < best_val_loss):
                best_val_loss = eval_loss['val']
                utils.save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, config, probe_cluster)
        
        # Get batch
        timer.start('data_prep')
        X, Y, probe_targets = get_batch_bound('train')
        timer.end('data_prep')

        # Print sample of first batch (only once at the start of training)
        if iter_num == 0 and config.debug_data:
            utils.debug_print_batch(X, Y, probe_targets)
        
        # Forward pass
        timer.start('training_step')
        loss = utils.run_single_training_step(
            model, probe_cluster, optimizer, scaler, 
            X, Y, probe_targets, config, ctx, iter_num, get_batch_bound
        )
        timer.end('training_step')
        
        # Calculate timing
        iter_time = timer.end('iteration')
        
        # Log progress
        if iter_num % config.log_interval == 0:
            print(f"iter {iter_num}: model loss {loss:.4f}, time {iter_time*1000:.2f}ms")
            
            # Show timing breakdown
            if iter_num > 0:
                timer.summarize(iter_num)
                if config.wandb_log:
                    wandb.log({
                        "iteration": iter_num,
                        "train/step_loss": loss,
                        "train/step_time": iter_time*1000
                    })
        
        iter_num += 1

        if stop_requested[0]:
            print(f"Stopping training at iter {iter_num}")
            if config.show_final_eval_on_stop:
                eval_loss, probe_losses = utils.run_evaluation(model, probe_cluster, config, ctx, get_batch_bound, timer)
                print(f"Final evaluation: train loss {eval_loss['train']:.4f}, val loss {eval_loss['val']:.4f}")
            break
        
        # Early stopping for debugging
        if iter_num >= 50 and 'PROFILE_RUN' in os.environ:
            print("Profile run complete, stopping early")
            break
    
    # End of training - print final timing summary
    timer.end('total_training')
    print("\nTraining complete. Final timing summary:")
    timer.summarize()

    print("Saving final checkpoint...")
    utils.save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, config, probe_cluster, final=True)

    # At the end of your main function, right before wandb.finish()
    if config.wandb_log and not stop_requested[0]:
        # Log total training time as a metric
        total_training_time = timer.timings['total_training'][-1]
        wandb.run.summary["total_time_seconds"] = total_training_time
        wandb.run.summary["total_time_minutes"] = total_training_time / 60
        wandb.run.summary["total_time_hours"] = total_training_time / 3600
        
        # Log average iteration time
        iterations_completed = iter_num - 1  # Subtract 1 if iter_num starts at 0
        wandb.run.summary["avg_iteration_time_ms"] = (total_training_time * 1000) / iterations_completed

    wandb.finish()

if __name__ == "__main__":
    main()