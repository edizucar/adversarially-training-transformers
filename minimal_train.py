#!/usr/bin/env python

import os
import torch
import wandb
import signal
import argparse

from src.model import GPTConfig, GPT
from src.config import TrainingConfig
from src.probes import ProbeCluster
from src.utils import (
    TimingTracker, get_batch, in_quotes_feature, auto_tune_batch_size, odd_quotes_in_tokens,
    estimate_loss, estimate_probe_loss, setup_environment, load_dataset,
    get_vocab_size, load_model_from_checkpoint, initialize_probes,
    setup_training_components, create_batch_getter, update_learning_rate,
    save_checkpoint, setup_wandb, log_evaluation_results, debug_print_batch,
    run_single_training_step, run_evaluation, setup_signal_handler
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('config', type=str, help='Path to config file (YAML or Python)')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
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
    
    # Initialize timing tracker
    timer = TimingTracker()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    config.device = device
    
    config, ctx, ptdtype = setup_environment(config, device_type)
    
    # Load dataset
    data_dir = os.path.join('data', config.dataset)
    train_data, val_data = load_dataset(data_dir)
    
    # Get vocab size and maybe checkpoint
    vocab_size, checkpoint = get_vocab_size(config, data_dir, device)
    
    # Create model
    model_args = GPTConfig(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias if hasattr(config, 'bias') else False,
        vocab_size=vocab_size,  # Use vocab size from checkpoint or meta
        dropout=config.dropout,
    )
    model = GPT(model_args)
    model.to(device)

    # Load model from checkpoint if needed
    iter_num = 0
    best_val_loss = float('inf')

    if config.init_from == 'resume' and checkpoint is not None:
        model, iter_num, best_val_loss = load_model_from_checkpoint(model, checkpoint)
    
    # Intialize probes if needed
    if config.train_probes:
        probe_cluster = initialize_probes(config, device, checkpoint, ProbeCluster)
    
    # Setup training components
    optimizer, scaler = setup_training_components(model, config, device_type)
    
    # Create batch getter
    get_batch_bound = create_batch_getter(train_data, val_data, config, device, timer, get_batch)
    
    # Tune batch size if requested
    if config.auto_tune_batch_size:
        config.batch_size = auto_tune_batch_size(
            model, config, ctx, optimizer, get_batch_bound, probe_cluster
        )

    if config.wandb_log:
        setup_wandb(config)

    stop_requested, signal_handler = setup_signal_handler()
    signal.signal(signal.SIGINT, signal_handler)

    # Main training loop
    print("\nStarting training...")
    
    # Track total iteration time
    timer.start('total_training')
    
    while iter_num < config.max_iters:
        timer.start('iteration')
        
        # Learning rate decay
        timer.start('lr_update')
        lr = update_learning_rate(optimizer, config, iter_num)
        timer.end('lr_update')

        if iter_num > 0 and iter_num % config.eval_interval == 0:
            eval_loss, probe_losses = run_evaluation(model, probe_cluster, config, ctx, get_batch_bound, timer)

            log_evaluation_results(eval_loss, probe_losses, iter_num, lr, config)

            # Save checkpoint if needed
            if (not config.never_save_checkpoint and eval_loss['val'] < best_val_loss) or iter_num > config.max_iters:
                best_val_loss = eval_loss['val']
                save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, config, probe_cluster)
        
        # Get batch
        timer.start('data_prep')
        X, Y, probe_targets = get_batch_bound('train')
        timer.end('data_prep')

        # Print sample of first batch (only once at the start of training)
        if iter_num == 0 and config.debug_data:
            debug_print_batch(X, Y, probe_targets)
        
        # Forward pass
        timer.start('training_step')
        loss = run_single_training_step(
            model, probe_cluster, optimizer, scaler, 
            X, Y, probe_targets, config, ctx, iter_num, get_batch_bound
        )
        timer.end('training_step')
        
        # Calculate timing
        iter_time = timer.end('iteration')
        
        # Log progress
        if iter_num % config.log_interval == 0:
            print(f"iter {iter_num}: loss {loss:.4f}, time {iter_time*1000:.2f}ms")
            
            # Show timing breakdown every 20 iterations
            if iter_num > 0:
                timer.summarize(n_iterations=20)
                if config.wandb_log:
                    wandb.log({
                        "iteration": iter_num,
                        "train/step_loss": loss,
                        "train/step_time": iter_time*1000
                    })
        
        iter_num += 1

        if stop_requested[0]:
            if config.show_final_eval_on_stop:
                eval_loss, probe_losses = run_evaluation(model, probe_cluster, config, ctx, get_batch_bound, timer)
                print(f"Final evaluation: train loss {eval_loss['train']:.4f}, val loss {eval_loss['val']:.4f}")
            print("Saving final checkpoint...")
            save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, config, probe_cluster, final=True)
            if config.wandb_log:
                wandb.finish()
            break
        
        # Early stopping for debugging
        if iter_num >= 50 and 'PROFILE_RUN' in os.environ:
            print("Profile run complete, stopping early")
            break
    
    # End of training - print final timing summary
    timer.end('total_training')
    print("\nTraining complete. Final timing summary:")
    timer.summarize()

if __name__ == "__main__":
    main()