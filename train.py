#!/usr/bin/env python
"""
Training script for GPT model with probe interventions.
"""

import os
import time
import math
import signal
import argparse
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.model import GPTConfig, GPT
import src.ProbeIntervention as ProbeIntervention
from src.config import TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('config', type=str, nargs='?', help='Path to config file (YAML or Python)')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--show-final-eval-on-stop', action='store_true', 
                        help='Show final evaluation when stopping with Ctrl+C')
    return parser.parse_args()

def setup_distributed(config):
    """Set up distributed training if enabled"""
    if int(os.environ.get('RANK', -1)) == -1:
        # Not in distributed mode
        return False, 0, 0, 1, True, 0, config.gradient_accumulation_steps, config.device
    
    # Initialize distributed process group
    init_process_group(backend=config.backend)
    
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    # master_process = ddp_rank == 0
    master_process = True
    seed_offset = ddp_rank
    
    # Set CUDA device
    torch.cuda.set_device(device)
    
    # Scale down gradient accumulation steps proportionally
    grad_accum_steps = config.gradient_accumulation_steps
    assert grad_accum_steps % ddp_world_size == 0
    grad_accum_steps = grad_accum_steps // ddp_world_size
    
    return True, ddp_rank, ddp_local_rank, ddp_world_size, master_process, seed_offset, grad_accum_steps, device

def setup_pytorch(config, device_type):
    """Improved setup with better mixed precision defaults"""
    torch.manual_seed(1337 + config.seed_offset)
    
    # Enable TF32 precision for better performance on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # No need to reassign memory functions - use the built-in ones directly
    
    # Setup dtype and autocast context
    if config.dtype == 'auto':
        if device_type == 'cuda':
            if torch.cuda.is_bf16_supported():
                ptdtype = torch.bfloat16
            else:
                ptdtype = torch.float16
        else:
            ptdtype = torch.float32
    else:
        ptdtype = {
            'float32': torch.float32, 
            'bfloat16': torch.bfloat16, 
            'float16': torch.float16
        }[config.dtype]
    
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # For specific GPU types (add near model initialization)
    if device_type == 'cuda' and torch.cuda.get_device_properties(0).major >= 8:  # Ampere or newer
        torch.set_float32_matmul_precision('high')
    
    # Optimize memory allocator behavior
    torch.cuda.memory_stats(device=0)
    torch.cuda.reset_peak_memory_stats(device=0)
    
    return ctx, ptdtype

def get_batch(split, train_data, val_data, config, device, device_type):
    """Get a batch of data for training or validation"""
    data = train_data if split == 'train' else val_data
    
    # Random offsets for batch
    offsets = torch.randint(len(data) - config.block_size, (config.batch_size,))
    
    # Get input sequence
    x = torch.stack([torch.from_numpy((data[offset:offset+config.block_size]).astype(np.int64)) 
                    for offset in offsets])
    
    # Generate probe targets
    _, _, probe_targets = ProbeIntervention.in_quotes_feature_optimized(x)
    
    # Get target sequence (shifted by 1)
    y = torch.stack([torch.from_numpy((data[offset+1:offset+1+config.block_size]).astype(np.int64)) 
                    for offset in offsets])
    
    if device_type == 'cuda':
        # Pin arrays for async transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        probe_targets = probe_targets.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        probe_targets = probe_targets.to(device)
        
    return x, y, probe_targets

def get_batch_prefetched(split, train_data, val_data, config, device, device_type):
    """Prefetch and preprocess batches using CUDA streams"""
    data = train_data if split == 'train' else val_data
    
    # Create two streams for asynchronous data transfers
    if device_type == 'cuda':
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
    
    # Prefetch next batch while current batch is processing
    offsets = torch.randint(len(data) - config.block_size, (config.batch_size,))
    
    # Prepare input sequence
    with torch.cuda.stream(stream1) if device_type == 'cuda' else nullcontext():
        x = torch.stack([torch.from_numpy((data[offset:offset+config.block_size]).astype(np.int64)) 
                        for offset in offsets])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
    
    # Prepare target sequence in parallel
    with torch.cuda.stream(stream2) if device_type == 'cuda' else nullcontext():
        y = torch.stack([torch.from_numpy((data[offset+1:offset+1+config.block_size]).astype(np.int64)) 
                        for offset in offsets])
        if device_type == 'cuda':
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            y = y.to(device)
    
    # Generate probe targets on CPU first, then transfer to GPU
    x_cpu = torch.stack([torch.from_numpy((data[offset:offset+config.block_size]).astype(np.int64)) 
                        for offset in offsets])
    _, _, probe_targets = ProbeIntervention.in_quotes_feature_optimized(x_cpu)
    
    if device_type == 'cuda':
        probe_targets = probe_targets.pin_memory().to(device, non_blocking=True)
    else:
        probe_targets = probe_targets.to(device)
    
    # Synchronize streams before returning
    if device_type == 'cuda':
        torch.cuda.synchronize()
        
    return x, y, probe_targets

def get_batch_optimized(split, train_data, val_data, config, device, device_type):
    """Optimized batch preparation with minimal CPU-GPU synchronization"""
    data = train_data if split == 'train' else val_data
    
    # Generate all offsets at once
    offsets = torch.randint(len(data) - config.block_size, (config.batch_size,))
    
    # Prepare input and target sequences in a vectorized way
    x_np = np.stack([data[offset:offset+config.block_size] for offset in offsets])
    y_np = np.stack([data[offset+1:offset+1+config.block_size] for offset in offsets])
    
    # Convert to torch tensors
    x = torch.from_numpy(x_np.astype(np.int64))
    y = torch.from_numpy(y_np.astype(np.int64))
    
    # Generate probe targets more efficiently (in-quotes feature)
    # This is a placeholder - we'll optimize this function separately
    _, _, probe_targets = ProbeIntervention.in_quotes_feature_optimized(x)
    
    if device_type == 'cuda':
        # Pin memory for async transfer and move to device in one operation
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        probe_targets = probe_targets.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        probe_targets = probe_targets.to(device)
        
    return x, y, probe_targets

def init_model(config, meta_vocab_size, device):
    """Initialize or load the model"""
    model_args = {
        'n_layer': config.n_layer,
        'n_head': config.n_head, 
        'n_embd': config.n_embd,
        'block_size': config.block_size,
        'bias': config.bias,
        'vocab_size': None,
        'dropout': config.dropout
    }
    
    iter_num = 0
    best_val_loss = 1e9
    
    if config.init_from == 'scratch':
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
    elif config.init_from == 'resume':
        print(f"Resuming training from {config.source_dir}")
        ckpt_path = os.path.join(config.source_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Use checkpoint model args
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
            
        # Create and load model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        
        # Fix key prefixes if needed
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        model.load_state_dict(state_dict)
        # Only use checkpoint iter_num if we're actually continuing standard model training
        # (not when training adversarially against probes)
        if config.train_model and not config.train_adversarially:
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
        else:
            print("Starting a new training run (adversarial or probe-only), resetting iteration count to 0")
            iter_num = 0  # Reset iteration counter
            # Still keep the best validation loss to avoid regressing
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
        
    elif config.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
        override_args = dict(dropout=config.dropout)
        model = GPT.from_pretrained(config.init_from, override_args)
        
        # Read config params from loaded model
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
            
    # Crop block size if needed
    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)
        model_args['block_size'] = config.block_size
        
    # Move model to device
    model.to(device)
    return model, model_args, iter_num, best_val_loss

def init_probe_cluster(config, checkpoint=None):
    """Initialize the probe cluster"""

    probe_cluster = ProbeIntervention.ProbeCluster(
        number_of_probes=2*config.n_layer,
        probe_type=config.probe_type,
        input_dim=config.n_embd,
        output_dim=1,
        lr=config.probe_learning_rate,
        device=config.device
    )

    # Load probe weights if resuming
    if checkpoint is not None:
        try:
            probe_cluster.load_from_checkpoint(checkpoint, lr=config.probe_learning_rate, device=config.device)
            print("Successfully loaded probe weights from checkpoint")
        except (KeyError, AttributeError) as e:
            print("No probe weights found in checkpoint, initializing new probes")

    return probe_cluster

def get_lr_scheduler(config):
    """Return a learning rate scheduler function"""
    def lr_scheduler(iter_num):
        # Linear warmup
        if iter_num < config.warmup_iters:
            return config.learning_rate * iter_num / config.warmup_iters

        # Min learning rate after decay period
        if iter_num > config.lr_decay_iters:
            return config.min_lr

        # Cosine decay
        decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = config.min_lr + coeff * (config.learning_rate - config.min_lr)
        return lr
        
    return lr_scheduler

def estimate_loss(model, config, ctx, get_batch_fn):
    """Estimate loss on train and validation splits"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters, device=config.device)
        for k in range(config.eval_iters):
            X, Y, _ = get_batch_fn(split)
            with ctx:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.inference_mode()
def estimate_probe_loss(model, probe_cluster, config, ctx, get_batch_fn, display_probe_predictions=False):
    """Estimate loss and accuracy for all probes"""
    model.eval()
    probe_cluster.set_eval_mode()
    out = {'train':{}, 'val':{}}
    
    for split in ['train', 'val']:
        probe_losses = torch.zeros(config.eval_iters, probe_cluster.get_num_probes(), device=config.device)
        probe_accuracies = torch.zeros(config.eval_iters, probe_cluster.get_num_probes(), device=config.device)
        
        for k in range(config.eval_iters):
            X, Y, probe_targets = get_batch_fn(split) 
            with ctx:
                _, _, activations = model(X, Y)
            
            # Get losses for each probe
            l = [loss.item() for loss in probe_cluster.compute_probe_losses(activations, probe_targets)]
            probe_losses[k] = torch.tensor(l, device=config.device)
            
            # Get accuracies for each probe
            probe_accuracies[k] = torch.tensor(
                probe_cluster.compute_accuracies(activations, probe_targets),
                device=config.device
            )
        
        # Average over evaluation iterations
        averaged_probe_losses = probe_losses.mean(dim=0)
        averaged_probe_accuracies = probe_accuracies.mean(dim=0)       
        
        out[split]["loss"] = [averaged_probe_losses[i].item() for i in range(probe_cluster.get_num_probes())]
        out[split]["accuracy"] = [averaged_probe_accuracies[i].item() for i in range(probe_cluster.get_num_probes())]

    if display_probe_predictions:
        # Show an example of the probes in action in stdout
        probe_cluster.display_probe_predictions(
        """In the heart of an ancient forest, a wise old owl named Hoot perched on a gnarled branch, overlooking a meandering brook. Beside the water, a curious young fox approached and asked, "What's the secret to wisdom, old owl?" Hoot, puffing his feathers thoughtfully, replied, "True wisdom, young fox, comes from listening more to the world around you than you speak to it.\"""", 
        model, device=config.device)

    probe_cluster.set_train_mode()
    model.train()
    return out

@torch.inference_mode()
def probe_accuracy_sampling(model, probe_cluster, config, ctx, get_batch_fn, probe_num, num_batches=1):
    """Sample probe accuracies and return data for visualization"""
    data = []
    model.eval()
    probe_cluster.set_eval_mode()
    overall_acc = 0.0
    
    for batch_num in range(num_batches):
        X, Y, probe_targets = get_batch_fn("val")
        with ctx:
            _, _, activations = model(X, Y)
            b_text, b_tokens, b_feature_targets = ProbeIntervention.in_quotes_feature_optimized(X)
        
        for batch in range(config.batch_size):
            a = [a[batch].unsqueeze(dim=0) for a in activations]
            pt = probe_targets[batch].unsqueeze(dim=0)
            feature_predictions = probe_cluster.compute_probe_predictions(a)[probe_num]
            acc = probe_cluster.compute_accuracies(a, pt)[probe_num]

            text = b_text[batch]
            tokens = b_tokens[batch]
            feature_targets = b_feature_targets[batch]
            
            data.append((acc, {
                "text": text,
                "tokens": tokens,
                "feature_predictions": feature_predictions.squeeze(dim=0),
                "feature_targets": feature_targets,
                "feature_targets_2": probe_targets[batch]
            }))
            overall_acc += acc

    data.sort(key=lambda x: x[0])
    overall_acc /= len(data)
    
    model.train()
    probe_cluster.set_train_mode()
    return data, overall_acc

def process_probe_prediction_distribution(data, probe_num, display=True):
    """Process and display probe prediction distributions"""
    import matplotlib.pyplot as plt
    
    # Extract accuracy values
    numbers = [acc for acc, _ in data]

    # Plot histogram
    plt.figure()
    plt.hist(numbers, bins=10, range=(0, 1), edgecolor='black')
    
    # Add titles and labels
    plt.title('Accuracy of Probes (0 - 1)')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    
    # Ensure the graphs directory exists
    os.makedirs("graphs", exist_ok=True)
    
    # Save the figure
    plt.savefig(f"graphs/probe{probe_num}.png")
    plt.close()

    print("-" * 6 + "Probe prediction accuracy distribution" + "-" * 6)
    print("Worst Performer: ")
    print(f"acc = {data[0][0]}")
    
    # Display worst performer
    ProbeIntervention.display_features_from_tokens_and_feature_tensor(
       data[0][1]["tokens"],
       data[0][1]["feature_predictions"],
    )
    
    print("\nBest Performer: ")
    print(f"acc = {data[-1][0]}")
    
    # Display best performer
    ProbeIntervention.display_features_from_tokens_and_feature_tensor(
        data[-1][1]["tokens"],
        data[-1][1]["feature_predictions"],
    )
    print("\n")

def find_optimal_batch_size(model, config, ctx, get_batch_fn, probe_cluster=None, max_batch_size=512, step=32):
    """Find the largest batch size that fits in memory"""
    print("Finding optimal batch size...")
    
    original_batch_size = config.batch_size
    device = config.device
    
    # Try increasingly larger batch sizes
    for batch_size in range(original_batch_size, max_batch_size + step, step):
        config.batch_size = batch_size
        try:
            # Clear cache to get accurate measurement
            if 'cuda' in device:
                torch.cuda.empty_cache()
            
            # Try to process a batch
            X, Y, probe_targets = get_batch_fn('train')
            with ctx:
                logits, loss, activations = model(X, Y)
                if config.train_adversarially:
                    # Also test probe backward pass
                    probe_losses = probe_cluster.compute_probe_losses(activations, probe_targets)
                    adversarial_loss = -sum(probe_losses)
                    total_loss = loss + adversarial_loss
                    total_loss.backward()

                    # Clean up
                    model.zero_grad(set_to_none=True)
                    if hasattr(probe_cluster, 'zero_grad_probes'):
                        probe_cluster.zero_grad_probes()
            
            # If successful, continue to the next batch size
            print(f"Batch size {batch_size} works!")
            
            # Free memory
            del X, Y, probe_targets, logits, loss, activations
            if 'cuda' in device:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # Found the limit, revert to the last working batch size
                config.batch_size = max(original_batch_size, batch_size - step)
                print(f"CUDA OOM at batch size {batch_size}, reverting to {config.batch_size}")
                break
            else:
                raise
    
    print(f"Optimal batch size: {config.batch_size}")
    return config.batch_size

def adaptive_accumulation_steps(config):
    """Adjust gradient accumulation steps based on batch size"""
    if not hasattr(config, 'original_batch_size') or config.original_batch_size == config.batch_size:
        return config.gradient_accumulation_steps
        
    # Determine base computational budget (batch_size * gradient_accumulation_steps)
    computational_budget = config.original_batch_size * config.gradient_accumulation_steps
    
    # Adjust if batch size was auto-tuned
    new_steps = max(1, int(computational_budget / config.batch_size))
    print(f"Adjusted gradient accumulation steps from {config.gradient_accumulation_steps} to {new_steps}")
    return new_steps

class PerformanceMonitor:
    def __init__(self):
        self.history = {
            'iteration': [],
            'loss': [],
            'mfu': [],
            'batch_size': [],
            'throughput': [],  # tokens/second
            'memory_used': [],
        }
    
    def update(self, iter_num, loss, mfu, batch_size, tokens_per_sec):
        self.history['iteration'].append(iter_num)
        self.history['loss'].append(loss)
        self.history['mfu'].append(mfu)
        self.history['batch_size'].append(batch_size)
        self.history['throughput'].append(tokens_per_sec)
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.history['memory_used'].append(memory_used)
    
    def print_stats(self, window=10):
        """Print average stats for the last 'window' iterations"""
        if len(self.history['iteration']) < window:
            return
        
        recent_mfu = np.mean(self.history['mfu'][-window:]) * 100
        recent_throughput = np.mean(self.history['throughput'][-window:])
        
        print(f"Recent performance metrics (last {window} iterations):")
        print(f"  MFU: {recent_mfu:.2f}%")
        print(f"  Throughput: {recent_throughput:.2f} tokens/sec")
        
        if 'memory_used' in self.history and self.history['memory_used']:
            recent_memory = np.mean(self.history['memory_used'][-window:])
            print(f"  GPU Memory: {recent_memory:.2f} GB")

def main():
    args = parse_args()
    
    if args.config:
        # Load configuration
        if args.config.endswith('.yaml'):
            config = TrainingConfig.from_yaml(args.config)
        else:
            config = TrainingConfig.from_py(args.config)
    else:
        config = TrainingConfig()
        config.dataset = "tiny_stories"
        config.n_layer = 12
        config.n_head = 12
        config.n_embd = 768
        config.batch_size = 128
        config.block_size = 512
        config.dropout = 0.0
        config.learning_rate = 1e-3
        config.min_lr = 1e-4
        config.max_iters = 6000
        config.eval_interval = 1000
        config.lr_decay_iters = 6000
        config.init_from = 'scratch'
        config.probe_type = 'linear'
        config.probe_learning_rate = 1e-3
        config.source_dir = "checkpoints/default"
        config.dest_dir = "checkpoints/default"
        config.lambda_adversarial = 1e-3

    # Override config with command line args
    if args.resume:
        config.init_from = 'resume'
    if args.eval_only:
        config.eval_only = True
    if args.no_wandb:
        config.wandb_log = False
    if args.show_final_eval_on_stop:
        config.show_final_eval_on_stop = True
    
    # Set up directories
    if config.master_process:
        os.makedirs(config.source_dir, exist_ok=True)
        os.makedirs(config.dest_dir, exist_ok=True)
    
    # Set up distributed training or single GPU
    (config.is_distributed, config.rank, config.local_rank, 
     config.world_size, config.master_process, config.seed_offset, 
     config.gradient_accumulation_steps, config.device) = setup_distributed(config)
     
    # Calculate tokens per iteration
    config.tokens_per_iter = (config.gradient_accumulation_steps * config.world_size * 
                             config.batch_size * config.block_size)
    
    # Setup PyTorch settings
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    ctx, ptdtype = setup_pytorch(config, device_type)
    
    # Load dataset
    data_dir = os.path.join('data', config.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Get batch function with config bound to it
    def get_batch_bound(split):
        return get_batch_prefetched(split, train_data, val_data, config, config.device, device_type)
    
    # Get vocabulary size from meta.pkl if available
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"Found vocab_size = {meta_vocab_size} (from {meta_path})")
    
    # Initialize model
    model, model_args, iter_num, best_val_loss = init_model(config, meta_vocab_size, config.device)
    
    # Initialize probe cluster
    checkpoint = None
    if config.init_from == 'resume':
        ckpt_path = os.path.join(config.source_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=config.device)
    probe_cluster = init_probe_cluster(config, checkpoint)

    if config.auto_tune_batch_size and torch.cuda.is_available():
        config.batch_size = find_optimal_batch_size(model, config, ctx, get_batch_bound, probe_cluster)
    
    if hasattr(config, 'original_batch_size') and config.batch_size != config.original_batch_size:
        config.gradient_accumulation_steps = adaptive_accumulation_steps(config)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        config.weight_decay, 
        config.learning_rate, 
        (config.beta1, config.beta2), 
        device_type
    )
    
    if config.init_from == 'resume' and checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Compile model if requested
    if config.compile:
        print("Compiling model (takes a ~minute)...")
        model = torch.compile(model)
    
    # Wrap model in DDP if distributed
    if config.is_distributed:
        model = DDP(model, device_ids=[config.local_rank])
    
    # Get raw model (unwrap DDP)
    raw_model = model.module if config.is_distributed else model
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    # Get learning rate scheduler
    lr_scheduler = get_lr_scheduler(config)
    
    # Set up wandb logging
    if config.wandb_log and config.master_process:
        import wandb
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))
        
        # Define metrics
        wandb.define_metric("iteration")
        wandb.define_metric("train/*", step_metric="iteration")
        wandb.define_metric("val/*", step_metric="iteration")
    
    # Set up signal handler for graceful exit
    stop_requested = False
    def signal_handler(sig, frame):
        nonlocal stop_requested
        print("Ctrl+C pressed. Finishing current task...")
        stop_requested = True
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get initial batch
    X, Y, probe_targets = get_batch_bound('train')
    
    # Training loop
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    performance_monitor = PerformanceMonitor()
    
    # At the start of the training loop:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels for your hardware
    
    while True:
        # Set learning rate for this iteration
        lr = lr_scheduler(iter_num) if config.decay_lr else config.learning_rate
        if config.train_model and config.decay_lr:
            # Only use scheduler if we're actually training the model
            lr = lr_scheduler(iter_num) 
        else:
            # Otherwise just use a fixed value for logging purposes
            lr = config.learning_rate if config.train_model else 0.0
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate and checkpoint
        if iter_num % config.eval_interval == 0 and config.master_process:
            losses = estimate_loss(model, config, ctx, get_batch_bound) if config.train_model else {"train": 10.0, "val": 10.0}
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Prepare a single log_dict for wandb
            if config.wandb_log:
                log_dict = {
                    "iteration": iter_num,
                    "train/model_loss": losses['train'],
                    "val/model_loss": losses['val'],
                    "model_lr": lr,
                    "probe_lr": config.probe_learning_rate,
                    "mfu": running_mfu * 100  # convert to percentage
                }

            # Add probe evaluation
            if config.train_probes:
                probe_eval_stats = estimate_probe_loss(model, probe_cluster, config, ctx, get_batch_bound)
                print("Probe losses and accuracies:")
                for split in ['train', 'val']:
                    for stat in probe_eval_stats[split]:
                        for i in range(probe_cluster.get_num_probes()):
                            probe_type = 'attn' if i % 2 == 0 else 'MLP'
                            layer = i // 2
                            print(f"{split} probe {i} ({probe_type}-{layer}) {stat}: {probe_eval_stats[split][stat][i]:.4f}")
                            
                # Add probe stats to wandb log_dict
                if config.wandb_log:
                    for split in probe_eval_stats:
                        for stat in probe_eval_stats[split]:
                            for i in range(probe_cluster.get_num_probes()):
                                log_dict[f"{split}/probe-{stat}-{'attn' if i%2==0 else 'MLP'}-layer-{i//2}"] = probe_eval_stats[split][stat][i]
            
            # Log to wandb (only once)
            if config.wandb_log:
                wandb.log(log_dict)
            
            # Save checkpoint if needed
            if ((not config.never_save_checkpoint and losses['val'] < best_val_loss) or 
                config.always_save_checkpoint or iter_num > config.max_iters):
                
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(config),
                    }
                    
                    # Add probe information
                    checkpoint.update(probe_cluster.state_dict())
                    
                    print(f"Saving checkpoint to {config.dest_dir}")
                    torch.save(checkpoint, os.path.join(config.dest_dir, 'ckpt.pt'))
        
        # Exit if evaluation only
        if iter_num == 0 and config.eval_only:
            break
        
        # Training steps
        for micro_step in range(config.gradient_accumulation_steps):
            if config.is_distributed:
                model.require_backward_grad_sync = (
                    micro_step == config.gradient_accumulation_steps - 1
                )
            
            # REPLACE FROM HERE
            # X, Y, probe_targets = get_batch_bound('train')
            
            # # Add this line here - right before entering the context manager
            # torch.cuda.set_stream(torch.cuda.Stream())
            
            # with ctx:
            #     # Use gradient checkpointing if enabled
            #     logits, transformer_loss, activations = model(X, Y, use_checkpoint=config.use_gradient_checkpointing)
            #     # Scale loss for gradient accumulation
            #     total_loss = transformer_loss / config.gradient_accumulation_steps
            
            # # Process probes and adversarial training
            # if config.train_probes:
            #     # Improved approach - batched processing
            #     BATCH_SIZE = min(4, config.phi_probe_steps_per_model_update)

            #     if BATCH_SIZE > 1:
            #         # Prepare batches in parallel
            #         all_X = []
            #         all_Y = []
            #         all_probe_targets = []
                    
            #         # Get all batches first (better memory locality)
            #         for _ in range(BATCH_SIZE):
            #             X_new, Y_new, probe_targets_new = get_batch_bound('train')
            #             all_X.append(X_new)
            #             all_Y.append(Y_new)
            #             all_probe_targets.append(probe_targets_new)
                    
            #         # Process batches with a single larger forward pass
            #         # Concatenate on batch dimension
            #         batched_X = torch.cat(all_X, dim=0)
            #         batched_Y = torch.cat(all_Y, dim=0)
                    
            #         # Single larger forward pass instead of multiple small ones
            #         with ctx:
            #             _, _, batched_activations = model(batched_X, batched_Y, use_checkpoint=False)
                        
            #         # Split activations back into original batch size chunks
            #         split_size = batched_X.size(0) // BATCH_SIZE
                    
            #         for step in range(BATCH_SIZE):
            #             # Extract slice of activations for this probe update
            #             start_idx = step * split_size
            #             end_idx = (step + 1) * split_size
                        
            #             # Extract activations for this batch
            #             step_activations = [
            #                 act[start_idx:end_idx].detach() 
            #                 for act in batched_activations
            #             ]
                        
            #             # Train probes on this batch
            #             probe_cluster.loss_backward_probes(step_activations, all_probe_targets[step], scaler)
            #             probe_cluster.optimiser_step_probes(scaler)
            #             probe_cluster.zero_grad_probes()
                
            #     # Adversarial training
            #     if config.train_adversarially:
            #         with ctx:
            #             # Compute probe losses on fresh activations (not detached)
            #             probe_losses = probe_cluster.compute_probe_losses(activations, probe_targets)
                        
            #             # Sum negative probe losses (adversarial objective)
            #             accumulated_adversarial_loss = torch.zeros_like(total_loss, device=config.device, dtype=torch.float)
            #             for probe_loss in probe_losses:
            #                 accumulated_adversarial_loss -= probe_loss
                        
            #             # Scale adversarial loss dynamically, avoiding division by zero
            #             if accumulated_adversarial_loss.abs().item() > 1e-10:
            #                 dynamic_scale = abs(config.lambda_adversarial * total_loss.detach() / 
            #                                 (accumulated_adversarial_loss.detach() + 1e-10))
            #                 weighted_adversarial_loss = dynamic_scale * accumulated_adversarial_loss
            #                 total_loss += weighted_adversarial_loss
            
            # # Backward pass
            # if config.train_model:
            #     scaler.scale(total_loss).backward()
            # REPLACE WITH:
            
            with torch.cuda.amp.autocast(enabled=(config.dtype in ['float16', 'bfloat16'])):
                # Get batch only once per step
                X, Y, probe_targets = get_batch_bound('train')
                
                # Forward pass with all computation needed for this step
                logits, transformer_loss, activations = model(X, Y, use_checkpoint=config.use_gradient_checkpointing)
                total_loss = transformer_loss / config.gradient_accumulation_steps
                
                # Process probes more efficiently
                if config.train_probes:
                    # Get detached activations for probe training
                    with torch.no_grad():
                        detached_activations = [act.detach() for act in activations]
                    
                    # Update probes multiple times per model update
                    # num_probe_updates = min(config.phi_probe_steps_per_model_update, 4)  # Cap at 4 for efficiency
                    num_probe_updates = config.phi_probe_steps_per_model_update

                    if num_probe_updates > 1:
                        # Multiple probe updates - get additional batches
                        for _ in range(num_probe_updates):
                            # Train probes on current batch
                            probe_cluster.loss_backward_probes(detached_activations, probe_targets, scaler)
                            probe_cluster.optimiser_step_probes(scaler)
                            
                            # Get a new batch for next probe update if not the last one
                            if _ < num_probe_updates - 1:
                                X_new, Y_new, probe_targets_new = get_batch_bound('train')
                                with torch.no_grad():
                                    # Get new activations without recomputing gradients
                                    _, _, new_activations = model(X_new, Y_new, use_checkpoint=False)
                                    detached_activations = [act.detach() for act in new_activations]
                                probe_targets = probe_targets_new
                    else:
                        # Just one probe update
                        probe_cluster.loss_backward_probes(detached_activations, probe_targets, scaler)
                        probe_cluster.optimiser_step_probes(scaler)
                    
                    # Adversarial training
                    if config.train_adversarially:
                        # Compute probe losses directly on the activations (not detached)
                        probe_losses = probe_cluster.compute_probe_losses(activations, probe_targets)
                        
                        # More numerically stable adversarial loss calculation
                        adversarial_loss = -sum(probe_losses) * config.lambda_adversarial
                        total_loss += adversarial_loss
                
                # Scale and backward
                if config.train_model:
                    scaler.scale(total_loss).backward()
        
        # Gradient clipping
        if config.grad_clip != 0.0 and config.train_model:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        if config.train_model:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config.log_interval == 0 and config.master_process:
            lossf = transformer_loss.item() * config.gradient_accumulation_steps
            
            if local_iter_num >= 5:  # Let training loop settle
                mfu = raw_model.estimate_mfu(
                    config.batch_size * config.gradient_accumulation_steps, dt
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                # Update performance monitor
                tokens_per_sec = config.batch_size * config.block_size / dt
                performance_monitor.update(iter_num, lossf, mfu, config.batch_size, tokens_per_sec)
                
                # Print stats every 10 iterations
                if iter_num % (config.log_interval * 10) == 0:
                    performance_monitor.print_stats()
                
            print(f"Iter {iter_num}, loss {lossf:.4f}, lr {lr}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        # Increment counters
        iter_num += 1
        local_iter_num += 1
        
        # Check termination conditions
        if iter_num > config.max_iters or stop_requested:
            if stop_requested and not hasattr(config, 'show_final_eval_on_stop'):
                print("\nSkipping final evaluation due to interrupt and --show-final-eval-on-stop flag not set")
            else:
                print("Final iteration reached!")
                print("\nDisplay final information:")
                # Display final probe stats
                if config.train_probes:
                    # Evaluate probes with predictions display
                    probe_eval_stats = estimate_probe_loss(
                        model, probe_cluster, config, ctx, get_batch_bound, 
                        display_probe_predictions=True
                    )
                    
                    # Sample and display probe accuracy distributions
                    table = ""
                    for probe_num in range(config.n_layer * 2):
                        data, acc = probe_accuracy_sampling(
                            model, probe_cluster, config, ctx, get_batch_bound,
                            probe_num=probe_num, num_batches=20
                        )
                        table += f"Probe {probe_num}: {acc:.4f}\n"
                        
                        # Process and display last probe in full detail
                        is_last_probe = (probe_num == config.n_layer * 2 - 1)
                        process_probe_prediction_distribution(data, probe_num=probe_num, display=is_last_probe)
                    
                    print("Probe Accuracy Summary:")
                    print(table)
            
            print("Training complete!")
            break
        
        # Update performance monitor
        performance_monitor.update(iter_num, transformer_loss.item(), running_mfu, config.batch_size, config.tokens_per_iter / dt)
        
        if iter_num % 100 == 0 and torch.cuda.is_available():
            print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        # Add near the end of each iteration
        if torch.cuda.is_available() and iter_num % 50 == 0:
            torch.cuda.empty_cache()
    
    # Clean up
    if config.is_distributed:
        destroy_process_group()

    # Print final performance stats
    performance_monitor.print_stats()
    break

if __name__ == "__main__":
    main()