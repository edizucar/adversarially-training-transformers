import torch
import tiktoken
from typing import NamedTuple
from collections import defaultdict
import time
GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")

# Binary feature extractor output type
BinaryFeatureExtractorOutput = NamedTuple(
    "FeatureExtractorOutput",
    [
        ("text", list[list[str]]),  # batched (hence nested)
        ("tokens", torch.Tensor),   # shape [batch seq_len]
        ("features", torch.Tensor), # shape [batch seq_len]
    ],
)

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

def fast_in_quotes_feature(tokens, tokenizer=GPT2_TOKENIZER):
    """Ultra-fast approximation of quote detection using pure tensor operations"""
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Pre-allocate on device
    features = torch.zeros((batch_size, seq_len), device=device)
    
    # Find quote token ID (34 in GPT2 tokenizer)
    quote_token_id = 34  # " character token ID in GPT2
    quote_positions = (tokens == quote_token_id)

    # Vectorized version that avoids loops
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
    
    return BinaryFeatureExtractorOutput(
        text=None,  # Skip text decoding for performance
        tokens=tokens,
        features=features,
    )

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