#!/usr/bin/env python

import os
import time
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
import torch.profiler as profiler
import tiktoken
from collections import defaultdict

from src.model import GPTConfig, GPT
from src.config import TrainingConfig
from src.probes import ProbeCluster
from src.utils import TimingTracker, in_quotes_feature, auto_tune_batch_size, odd_quotes_in_tokens

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('config', type=str, help='Path to config file (YAML or Python)')
    return parser.parse_args()

def get_batch(split, train_data, val_data, config, device, timer):
    """Optimized batch preparation with minimal overhead"""
    data = train_data if split == 'train' else val_data
    
    # Create single tensor and fill it - more efficient than individual conversions
    x = torch.zeros((config.batch_size, config.block_size), dtype=torch.long, device='cpu')
    y = torch.zeros((config.batch_size, config.block_size), dtype=torch.long, device='cpu')

    timer.start('get_batch_loop')
    for i in range(config.batch_size):
        timer.start('get_batch_sample')
        valid = False
        attempts = 0
        max_attempts = 10
        while not valid and attempts < max_attempts:
            idx = torch.randint(len(data) - config.block_size, (1,)).item()
            tokens = data[idx:idx+config.block_size]
            if not odd_quotes_in_tokens(tokens):
                x[i] = torch.from_numpy(tokens.astype(np.int64))
                y[i] = torch.from_numpy(data[idx+1:idx+1+config.block_size].astype(np.int64))
                valid = True
            attempts += 1
        if not valid:
            x[i] = torch.from_numpy(tokens.astype(np.int64))
            y[i] = torch.from_numpy(data[idx+1:idx+1+config.block_size].astype(np.int64))
        timer.end('get_batch_sample')
    timer.end('get_batch_loop')

    # Move to device (single transfer)
    timer.start('get_batch_to_device')
    x, y = x.to(device), y.to(device)
    timer.end('get_batch_to_device')

    # Compute probe targets
    timer.start('get_batch_probe_targets')
    probe_targets = in_quotes_feature(x).features
    timer.end('get_batch_probe_targets')

    return x, y, probe_targets

def main():
    # Basic setup
    args = parse_args()
    if args.config.endswith('.yaml'):
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig.from_py(args.config)
    
    # Initialize timing tracker
    timer = TimingTracker()

    if not hasattr(config, 'gradient_accumulation_steps'):
        config.gradient_accumulation_steps = 1
    if not hasattr(config, 'phi_probe_steps_per_model_update'):
        config.phi_probe_steps_per_model_update = 1
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    config.device = device
    
    # Enable tensor cores and memory setting for reduced fragmentation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Setup dtype
    if config.dtype == 'auto':
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            config.dtype = 'bfloat16'
        elif torch.cuda.is_available():
            config.dtype = 'float16'
        else:
            config.dtype = 'float32'
    
    if config.dtype == 'float16':
        ptdtype = torch.float16
    elif config.dtype == 'bfloat16':
        ptdtype = torch.bfloat16
    else:
        ptdtype = torch.float32
    
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load dataset
    data_dir = os.path.join('data', config.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Load checkpoint FIRST to get model parameters
    if config.init_from == 'resume':
        ckpt_path = os.path.join(config.source_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Extract model_args from checkpoint or use checkpoint shape info
        if 'model_args' in checkpoint:
            model_args_dict = checkpoint['model_args']
            vocab_size = model_args_dict.get('vocab_size', None)
        else:
            # If no model_args, try to infer from weights
            for k in checkpoint['model'].keys():
                if k == 'transformer.wte.weight' or k == '_orig_mod.transformer.wte.weight':
                    vocab_size = checkpoint['model'][k].shape[0]
                    print(f"Inferred vocab_size = {vocab_size} from checkpoint weights")
                    break
    else:
        # Find vocab size from meta.pkl
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            import pickle
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            vocab_size = meta['vocab_size']
        else:
            vocab_size = None
    
    # Initialize model with correct vocab size
    model_args = GPTConfig(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias if hasattr(config, 'bias') else False,
        vocab_size=vocab_size,  # Use vocab size from checkpoint or meta
        dropout=config.dropout,
    )
    
    # Create model
    model = GPT(model_args)
    model.to(device)
    
    # Load parameters from checkpoint
    if config.init_from == 'resume':
        # Fix key prefixes if needed
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        # Load the state dict now that the model has the right dimensions
        model.load_state_dict(state_dict)
        
        # Get iteration number from checkpoint
        iter_num = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        iter_num = 0
        best_val_loss = float('inf')
    
    # Create probe cluster if needed
    if config.train_probes:
        probe_cluster = ProbeCluster(
            n_layer=config.n_layer,
            d_model=config.n_embd,
            learning_rate=config.probe_learning_rate,
            device=device
        )

        # Load probe cluster state from checkpoint if available
        if config.init_from == 'resume' and 'probe_state' in checkpoint:
            probe_cluster.load_state_dict(checkpoint['probe_state'])
    else:
        probe_cluster = None
    
    # Compile model if requested and available
    if config.compile and hasattr(torch, 'compile') and device_type == 'cuda':
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    
    # Configure optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type
    )
    
    # Create a local get_batch function bound to our data and config
    def get_batch_bound(split):
        timer.start('get_batch')
        batch = get_batch(split, train_data, val_data, config, device, timer)
        timer.end('get_batch')
        return batch
    
    # Create gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(config.dtype == 'float16'))
    
    # Tune batch size if requested
    if config.auto_tune_batch_size:
        config.batch_size = auto_tune_batch_size(
            model, config, ctx, optimizer, get_batch_bound, probe_cluster
        )
    
    # Main training loop
    print("\nStarting training...")
    iter_num = 0
    running_mfu = -1.0
    
    # Track total iteration time
    timer.start('total_training')
    
    while iter_num < config.max_iters:
        timer.start('iteration')
        
        # Learning rate decay
        timer.start('lr_update')
        if config.decay_lr:
            # Linear warmup then cosine decay
            if iter_num < config.warmup_iters:
                lr = config.learning_rate * iter_num / config.warmup_iters
            else:
                decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
                decay_ratio = min(decay_ratio, 1.0)
                lr = config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1.0 + math.cos(math.pi * decay_ratio))
            
            # Set learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * config.gradient_accumulation_steps
        timer.end('lr_update')
        
        # Zero gradients
        if iter_num == 0:
            timer.start('zero_grad')
            optimizer.zero_grad(set_to_none=True)
            if config.train_probes:
                probe_cluster.zero_grad_probes()
            timer.end('zero_grad')
        
        # Get batch
        timer.start('data_prep')
        X, Y, probe_targets = get_batch_bound('train')
        timer.end('data_prep')

        # Print sample of first batch (only once at the start of training)
        if iter_num == 0 and config.debug_data:
            enc = tiktoken.get_encoding("gpt2")
            print("\n=== Sample Batch Data ===")
            print(f"X shape: {X.shape}, Y shape: {Y.shape}, probe_targets shape: {probe_targets.shape}")
            print(f"\033[1mX sample:\033[0m\n{repr(enc.decode(X[0, :-1].tolist()))}")
            print(f"\033[1mY sample:\033[0m\n{repr(enc.decode(Y[0, :-1].tolist()))}")
            print(f"\033[1mprobe_targets sample:\033[0m\n{[(enc.decode([X[0, i].item()]), probe_targets[0, i].item()) for i in range(len(X[0, :-1]))]}".replace("), ", "),\n"))
            print("=========================\n")
        
        # Forward pass
        timer.start('forward_pass')
        with ctx:
            logits, transformer_loss, activations = model(X, Y, use_checkpoint=config.use_gradient_checkpointing)
            
            # Process probes and adversarial training
            timer.start('probe_computation')
            if config.train_probes and config.train_adversarially:
                # Compute probe losses directly
                probe_losses = probe_cluster.compute_probe_losses(activations, probe_targets)
                adversarial_loss = -sum(probe_losses) * config.lambda_adversarial
                total_loss = transformer_loss + adversarial_loss
            else:
                total_loss = transformer_loss
            timer.end('probe_computation')
        timer.end('forward_pass')
        
        # Backward pass
        timer.start('backward_pass')
        if config.train_model:
            scaler.scale(total_loss).backward()

            if (iter_num + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                timer.start('grad_clip')
                if config.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                timer.end('grad_clip')
            
                # Optimizer step
                timer.start('optimizer_step')
                scaler.step(optimizer)
                scaler.update()
                timer.end('optimizer_step')

                # Zero gradients
                optimizer.zero_grad(set_to_none=True)
                if config.train_probes:
                    probe_cluster.zero_grad_probes()
        timer.end('backward_pass')
        
        # Update probes based on phi_probe_steps_per_model_update
        timer.start('probe_update')
        if config.train_probes and config.train_adversarially:
            # Simple probe update with detached activations
            with torch.no_grad():
                detached_activations = [act.detach() for act in activations]
            
            # Update probes multiple times per model update
            for _ in range(config.phi_probe_steps_per_model_update):
                # Update probes with current batch
                probe_cluster.update_probes(
                    detached_activations,
                    probe_targets,
                    scaler=scaler if config.dtype == 'float16' else None
                )
                
                # Get a new batch for next probe update if not the last one
                if _ < config.phi_probe_steps_per_model_update - 1:
                    X_new, Y_new, probe_targets_new = get_batch_bound('train')
                    with torch.no_grad():
                        # Get new activations without recomputing gradients
                        _, _, new_activations = model(X_new, Y_new, use_checkpoint=config.use_gradient_checkpointing)
                        detached_activations = [act.detach() for act in new_activations]
                    probe_targets = probe_targets_new
        timer.end('probe_update')
        
        # Calculate timing
        iter_time = timer.end('iteration')
        
        # Log progress
        if iter_num % config.log_interval == 0:
            print(f"iter {iter_num}: loss {transformer_loss.item():.4f}, time {iter_time*1000:.2f}ms")
            
            # Show timing breakdown every 20 iterations
            if iter_num > 0 and iter_num % 20 == 0:
                timer.summarize(n_iterations=20)
        
        iter_num += 1
        
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