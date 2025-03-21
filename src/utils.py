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

def odd_quotes_in_text(text):
    """Find examples with an odd number of quotation marks."""
    quote_count = text.count('"')
    if quote_count % 2 != 0:
        return True
    return False

def num_quotes_in_token(token):
    """Return the number of quotation marks in a token using just its ID"""
    TOKENS_WITH_QUOTES = {
        1: 1,366: 1, 526: 1, 553: 1, 1298: 1, 1600: 1, 1701: 1, 1911: 1, 2404: 2, 2430: 2, 2474: 1, 2625: 1, 4895: 1, 4943: 1, 5320: 1, 5855: 1, 7203: 1, 7879: 1, 8172: 1, 8351: 2, 8762: 1,
        8973: 1, 9063: 1, 9313: 1, 9962: 1, 11074: 1, 11097: 1, 11496: 1, 11919: 2, 12340: 1, 12813: 1, 12878: 1, 13018: 2, 13538: 2, 13984: 1, 14631: 1, 14692: 1, 15327: 1, 15341: 1, 15473: 2,
        15931: 2, 16078: 1, 16725: 1, 17241: 1, 17553: 1, 17912: 1, 17971: 1, 18109: 1, 18161: 1, 19056: 1, 19570: 1, 19779: 1, 19990: 1, 20598: 1, 20662: 1, 21215: 1, 21387: 1, 22039: 1, 22135: 1,
        23785: 1, 23984: 1, 24018: 1, 24426: 1, 24618: 1, 25113: 1, 25698: 1, 25719: 1, 26033: 1, 26214: 1, 26358: 2, 26700: 1, 26793: 1, 26989: 1, 27071: 1, 27267: 1, 27444: 1, 27896: 1, 29225: 1,
        29368: 1, 29653: 1, 30478: 1, 30487: 1, 30543: 1, 30629: 1, 30823: 1, 30827: 1, 30866: 1, 32047: 1, 32203: 2, 32509: 2, 33116: 1, 33151: 2, 33172: 1, 33283: 1, 33490: 1, 34171: 2, 34607: 1,
        34713: 4, 35379: 1, 35713: 1, 35922: 1, 36521: 1, 36786: 1, 37082: 1, 37160: 1, 37227: 3, 37811: 3, 38214: 1, 39658: 1, 40264: 1, 40484: 1, 40754: 1, 41424: 2, 42501: 1, 42720: 1, 42785: 2,
        42911: 1, 42924: 1, 43634: 1, 43825: 1, 44212: 1, 44388: 1, 45144: 1, 45434: 1, 46385: 1, 47182: 4, 48219: 1, 48220: 1, 48774: 1, 49296: 1, 50248: 1
    }
    return TOKENS_WITH_QUOTES.get(token, 0)
    
def odd_quotes_in_tokens(tokens):
    """Find examples with an odd number of quotation marks."""
    TOKENS_WITH_QUOTES = {
        1: 1,366: 1, 526: 1, 553: 1, 1298: 1, 1600: 1, 1701: 1, 1911: 1, 2404: 2, 2430: 2, 2474: 1, 2625: 1, 4895: 1, 4943: 1, 5320: 1, 5855: 1, 7203: 1, 7879: 1, 8172: 1, 8351: 2, 8762: 1,
        8973: 1, 9063: 1, 9313: 1, 9962: 1, 11074: 1, 11097: 1, 11496: 1, 11919: 2, 12340: 1, 12813: 1, 12878: 1, 13018: 2, 13538: 2, 13984: 1, 14631: 1, 14692: 1, 15327: 1, 15341: 1, 15473: 2,
        15931: 2, 16078: 1, 16725: 1, 17241: 1, 17553: 1, 17912: 1, 17971: 1, 18109: 1, 18161: 1, 19056: 1, 19570: 1, 19779: 1, 19990: 1, 20598: 1, 20662: 1, 21215: 1, 21387: 1, 22039: 1, 22135: 1,
        23785: 1, 23984: 1, 24018: 1, 24426: 1, 24618: 1, 25113: 1, 25698: 1, 25719: 1, 26033: 1, 26214: 1, 26358: 2, 26700: 1, 26793: 1, 26989: 1, 27071: 1, 27267: 1, 27444: 1, 27896: 1, 29225: 1,
        29368: 1, 29653: 1, 30478: 1, 30487: 1, 30543: 1, 30629: 1, 30823: 1, 30827: 1, 30866: 1, 32047: 1, 32203: 2, 32509: 2, 33116: 1, 33151: 2, 33172: 1, 33283: 1, 33490: 1, 34171: 2, 34607: 1,
        34713: 4, 35379: 1, 35713: 1, 35922: 1, 36521: 1, 36786: 1, 37082: 1, 37160: 1, 37227: 3, 37811: 3, 38214: 1, 39658: 1, 40264: 1, 40484: 1, 40754: 1, 41424: 2, 42501: 1, 42720: 1, 42785: 2,
        42911: 1, 42924: 1, 43634: 1, 43825: 1, 44212: 1, 44388: 1, 45144: 1, 45434: 1, 46385: 1, 47182: 4, 48219: 1, 48220: 1, 48774: 1, 49296: 1, 50248: 1
    }
    quote_count = sum(TOKENS_WITH_QUOTES.get(token, 0) for token in tokens)
    if quote_count % 2 != 0:
        return True
    return False

def in_quotes_feature_old(
    tokens: torch.Tensor,
    tokenizer: any = GPT2_TOKENIZER
) -> BinaryFeatureExtractorOutput:
    """
    Detect tokens that are inside quotes or contain quotes.
    
    Args:
        tokens: Batched token IDs [batch_size, seq_len]
        tokenizer: The tokenizer used to encode the text
        
    Returns:
        BinaryFeatureExtractorOutput with features indicating tokens inside quotes and quote tokens
    """
    features = torch.zeros_like(tokens, dtype=torch.float, device=tokens.device)
    quote_counts = {}
    
    for b in range(tokens.shape[0]):
        in_quote = False
        
        for pos in range(tokens.shape[1]):
            token_id = tokens[b, pos].item()
            
            # Get/compute quote count (with lazy caching)
            if token_id not in quote_counts:
                quote_counts[token_id] = num_quotes_in_token(token_id)
            
            # Mark if inside quotes or contains quotes
            if in_quote or quote_counts[token_id] > 0:
                features[b, pos] = 1.0
            
            # Toggle quote state for each quote in token
            for _ in range(quote_counts[token_id]):
                in_quote = not in_quote
    
    return BinaryFeatureExtractorOutput(text=None, tokens=tokens, features=features)

def in_quotes_feature(
    tokens: torch.Tensor,
    tokenizer: any = GPT2_TOKENIZER
) -> BinaryFeatureExtractorOutput:
    """
    Detect tokens inside quotes using regex on the detokenized text.
    
    Args:
        tokens: Batched token IDs [batch_size, seq_len]
        tokenizer: The tokenizer used to encode the text
        
    Returns:
        BinaryFeatureExtractorOutput with features indicating tokens inside quotes
    """
    features = torch.zeros_like(tokens, dtype=torch.float, device=tokens.device)
    
    for b in range(tokens.shape[0]):
        # Convert tokens to text
        batch_tokens = tokens[b, :].tolist()
        text = tokenizer.decode(batch_tokens)
        
        # Find character positions of all quotes
        quote_positions = [i for i, char in enumerate(text) if char == '"']
        char_in_quotes = [False] * len(text)
        
        # Mark characters between quote pairs
        for i in range(0, len(quote_positions) - 1, 2):
            if i + 1 < len(quote_positions):
                start, end = quote_positions[i], quote_positions[i + 1]
                for j in range(start, end + 1):  # Include the quotes
                    char_in_quotes[j] = True
        
        # Map character positions back to tokens
        char_pos = 0
        for pos, token_id in enumerate(batch_tokens):
            token_text = tokenizer.decode([token_id])
            token_len = len(token_text)
            
            # Check if any character in this token's range is in quotes
            in_quotes = False
            for j in range(char_pos, min(char_pos + token_len, len(char_in_quotes))):
                if char_in_quotes[j]:
                    in_quotes = True
                    break
            
            if in_quotes:
                features[b, pos] = 1.0
            
            char_pos += token_len
    
    return BinaryFeatureExtractorOutput(text=None, tokens=tokens, features=features)

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