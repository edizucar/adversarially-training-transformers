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

class MinimalProbeCluster:
    def __init__(self, n_layer: int, d_model: int, learning_rate: float = 1e-3, device="cuda"):
        self.n_layer = n_layer
        self.d_model = d_model
        self.lr = learning_rate
        
        # Create probes
        self.probes = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_layer * 2)
        ]).to(device)
        
        # Create optimizers
        self.optimizers = [
            torch.optim.Adam(probe.parameters(), lr=learning_rate)
            for probe in self.probes
        ]
    
    def compute_probe_losses(self, activations, targets):
        """Compute BCE loss for each probe"""
        losses = []
        for act, probe in zip(activations, self.probes):
            logits = probe(act).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
            losses.append(loss)
        return losses
    
    def update_probes(self, activations, targets):
        """Update all probes in one go"""
        losses = self.compute_probe_losses(activations, targets)
        
        # Update each probe
        for i, (loss, opt) in enumerate(zip(losses, self.optimizers)):
            opt.zero_grad()
            if i < len(losses) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            opt.step()
        return losses
    
    def zero_grad_probes(self):
        """Zero gradients for all probe optimizers"""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

def fast_in_quotes_feature(tokens, device="cuda"):
    """Ultra-fast approximation of quote detection using pure tensor operations"""
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Pre-allocate on device
    features = torch.zeros((batch_size, seq_len), device=device)
    
    # Find quote token ID (34 in GPT2 tokenizer)
    quote_token_id = 34  # " character token ID in GPT2
    quote_positions = (tokens == quote_token_id)

    # Vectorized version that avoids the loop
    # Create cumulative sums for all sequences at once
    quote_counts = torch.cumsum(quote_positions.to(torch.float), dim=1)
    
    # Mask for valid positions (after first quote in each sequence)
    first_quote_pos = torch.argmax((quote_positions).to(torch.float), dim=1)
    valid_mask = torch.arange(seq_len, device=device).unsqueeze(0) > first_quote_pos.unsqueeze(1)
    
    # Apply the toggle using modulo operation (inside quotes = odd count)
    inside_quotes = (quote_counts % 2).to(torch.float)
    
    # Shift by 1 position (we want inside the quotes, not the quote marks themselves)
    shifted_inside = torch.zeros_like(inside_quotes, device=device)
    shifted_inside[:, 1:] = inside_quotes[:, :-1]
    
    # Apply valid mask - only count positions after first quote
    features = shifted_inside * valid_mask.to(torch.float)
    
    return features

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('config', type=str, help='Path to config file (YAML or Python)')
    return parser.parse_args()

def get_batch(split, train_data, val_data, config, device):
    """Optimized batch preparation with minimal overhead"""
    data = train_data if split == 'train' else val_data
    
    # Get random offsets for batch
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    
    # Create single tensor and fill it - more efficient than individual conversions
    x = torch.zeros((config.batch_size, config.block_size), dtype=torch.long, device='cpu')
    y = torch.zeros((config.batch_size, config.block_size), dtype=torch.long, device='cpu')
    
    # Fill tensors (still has a loop but minimizes conversions)
    for i, idx in enumerate(ix):
        x[i] = torch.from_numpy(data[idx:idx+config.block_size].astype(np.int64))
        y[i] = torch.from_numpy(data[idx+1:idx+1+config.block_size].astype(np.int64))
    
    # Move to device (single transfer)
    x, y = x.to(device), y.to(device)
    
    probe_targets = fast_in_quotes_feature(x)
    
    return x, y, probe_targets

# Add this timing utility class
class TimingTracker:
    def __init__(self):
        self.timings = defaultdict(list)
        self.active_timers = {}
        
    def start(self, name):
        """Start timing an operation"""
        self.active_timers[name] = time.time()
        
    def end(self, name):
        """End timing an operation and record the duration"""
        if name in self.active_timers:
            duration = time.time() - self.active_timers[name]
            self.timings[name].append(duration)
            del self.active_timers[name]
            return duration
        return None
    
    def summarize(self, n_iterations=None):
        """Print a summary of timing information"""
        print("\n=== TIMING SUMMARY ===")
        total_time = sum(sum(times) for times in self.timings.values())
        
        # Calculate statistics for each operation
        stats = {}
        for name, times in self.timings.items():
            if n_iterations is not None:
                # Only consider the last n_iterations
                times = times[-n_iterations:]
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            total_op_time = sum(times)
            percentage = (total_op_time / total_time) * 100 if total_time > 0 else 0
            
            stats[name] = {
                'avg_ms': avg_time * 1000,
                'max_ms': max_time * 1000,
                'min_ms': min_time * 1000,
                'total_s': total_op_time,
                'percentage': percentage
            }
        
        # Sort by total time (descending)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_s'], reverse=True)
        
        # Print table
        print(f"{'Operation':<30} {'Avg (ms)':<10} {'Max (ms)':<10} {'Min (ms)':<10} {'% of Total':<10}")
        print("-" * 70)
        for name, stat in sorted_stats:
            print(f"{name:<30} {stat['avg_ms']:<10.2f} {stat['max_ms']:<10.2f} {stat['min_ms']:<10.2f} {stat['percentage']:<10.2f}")
        
        print(f"\nTotal tracked time: {total_time:.2f} seconds")
        return stats

def auto_tune_batch_size(
    model, config, ctx, optimizer, get_batch_fn, probe_cluster=None
):
    """
    Automatically determine the optimal batch size for the current setup.
    Takes into account whether the model is compiled or not.
    
    Args:
        model: The PyTorch model
        config: Training configuration object
        ctx: The autocast context
        optimizer: The model optimizer
        get_batch_fn: Function to get a batch of data
        probe_cluster: Optional probe cluster for adversarial training
        
    Returns:
        int: The optimal batch size
    """
    print("Auto-tuning batch size...")
    # Store original batch size
    original_batch_size = config.batch_size
    # Set a smaller step factor when using compile
    step_factor = 2 if not config.compile else 1.25
    # Apply a safety factor for compiled models
    safety_factor = 0.9 if config.compile else 1
    # Keep track of the last successful batch size
    last_successful_size = config.batch_size
    
    try:
        # Start with the current batch size and try to increase it
        while True:
            # Try to process a batch with this size
            torch.cuda.empty_cache()
            X, Y, probe_targets = get_batch_fn('train')
            
            # Test complete forward and backward pass
            with ctx:
                logits, loss, activations = model(X, Y, use_checkpoint=config.use_gradient_checkpointing)
                
                # If adversarial training is enabled, test that too
                if config.train_probes and config.train_adversarially:
                    # Compute probe losses
                    if probe_cluster is not None:
                        probe_losses = probe_cluster.compute_probe_losses(activations, probe_targets)
                        adversarial_loss = -sum(probe_losses) * config.lambda_adversarial
                        loss = loss + adversarial_loss
                
                # Test backward pass too
                loss.backward()
            
            # Clean up
            optimizer.zero_grad(set_to_none=True)
            del X, Y, probe_targets, logits, loss, activations
            torch.cuda.empty_cache()
            
            # This size worked, remember it
            last_successful_size = config.batch_size
            print(f"Batch size {config.batch_size} works!")
            
            # Calculate next batch size to try
            next_size = int(config.batch_size * step_factor)
            
            # If we're not making meaningful progress, stop
            if next_size <= config.batch_size + 1:
                break
                
            config.batch_size = next_size
            
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
            # If we hit OOM, revert to the last working size with safety factor
            config.batch_size = max(original_batch_size, int(last_successful_size * safety_factor))
            print(f"Hit memory limit. Setting batch size to {config.batch_size} with safety factor")
        else:
            # Some other error occurred
            config.batch_size = original_batch_size
            print(f"Error during batch size tuning: {e}")
            print(f"Reverting to original batch size: {config.batch_size}")
    
    # Apply an additional safety margin for compiled models
    if config.compile:
        prev_size = config.batch_size
        config.batch_size = max(1, int(config.batch_size * 0.9))  # Additional 10% reduction
        if prev_size != config.batch_size:
            print(f"Applied compiled model safety margin: {prev_size} → {config.batch_size}")
    
    # Return the determined batch size
    return config.batch_size

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
        probe_cluster = MinimalProbeCluster(
            n_layer=config.n_layer,
            d_model=config.n_embd,
            learning_rate=config.probe_learning_rate,
            device=device
        )
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
        batch = get_batch(split, train_data, val_data, config, device)
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
        
        # Update probes occasionally
        timer.start('probe_update')
        if config.train_probes and not config.train_adversarially and iter_num % 10 == 0:
            # Simple probe update with detached activations
            with torch.no_grad():
                detached_activations = [act.detach() for act in activations]
            
            # Update probes
            probe_cluster.update_probes(detached_activations, probe_targets)
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