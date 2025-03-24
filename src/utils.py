import os
import math
import time
import wandb
import torch
import tiktoken
import numpy as np
from typing import NamedTuple
from datetime import datetime
from contextlib import nullcontext
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

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
        print(f"\n=== TIMING SUMMARY at {n_iterations} iterations ===")
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

def odd_quotes_in_text(text):
    """Find examples with an odd number of quotation marks."""
    quote_count = text.count('"')
    if quote_count % 2 != 0:
        return True
    return False
    
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

def in_quotes_feature(
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
    TOKENS_WITH_QUOTES = {
        1: 1,366: 1, 526: 1, 553: 1, 1298: 1, 1600: 1, 1701: 1, 1911: 1, 2404: 2, 2430: 2, 2474: 1, 2625: 1, 4895: 1, 4943: 1, 5320: 1, 5855: 1, 7203: 1, 7879: 1, 8172: 1, 8351: 2, 8762: 1,
        8973: 1, 9063: 1, 9313: 1, 9962: 1, 11074: 1, 11097: 1, 11496: 1, 11919: 2, 12340: 1, 12813: 1, 12878: 1, 13018: 2, 13538: 2, 13984: 1, 14631: 1, 14692: 1, 15327: 1, 15341: 1, 15473: 2,
        15931: 2, 16078: 1, 16725: 1, 17241: 1, 17553: 1, 17912: 1, 17971: 1, 18109: 1, 18161: 1, 19056: 1, 19570: 1, 19779: 1, 19990: 1, 20598: 1, 20662: 1, 21215: 1, 21387: 1, 22039: 1, 22135: 1,
        23785: 1, 23984: 1, 24018: 1, 24426: 1, 24618: 1, 25113: 1, 25698: 1, 25719: 1, 26033: 1, 26214: 1, 26358: 2, 26700: 1, 26793: 1, 26989: 1, 27071: 1, 27267: 1, 27444: 1, 27896: 1, 29225: 1,
        29368: 1, 29653: 1, 30478: 1, 30487: 1, 30543: 1, 30629: 1, 30823: 1, 30827: 1, 30866: 1, 32047: 1, 32203: 2, 32509: 2, 33116: 1, 33151: 2, 33172: 1, 33283: 1, 33490: 1, 34171: 2, 34607: 1,
        34713: 4, 35379: 1, 35713: 1, 35922: 1, 36521: 1, 36786: 1, 37082: 1, 37160: 1, 37227: 3, 37811: 3, 38214: 1, 39658: 1, 40264: 1, 40484: 1, 40754: 1, 41424: 2, 42501: 1, 42720: 1, 42785: 2,
        42911: 1, 42924: 1, 43634: 1, 43825: 1, 44212: 1, 44388: 1, 45144: 1, 45434: 1, 46385: 1, 47182: 4, 48219: 1, 48220: 1, 48774: 1, 49296: 1, 50248: 1
    }
    features = torch.zeros_like(tokens, dtype=torch.float, device=tokens.device)
    batch_size, seq_len = tokens.shape

    unique_tokens = torch.unique(tokens.cpu()).tolist()

    for b in range(batch_size):
        in_quote = False
        batch_tokens = tokens[b].cpu().tolist()

        for pos in range(seq_len):
            token_id = batch_tokens[pos]
            quote_count = TOKENS_WITH_QUOTES.get(token_id, 0)
            if in_quote or quote_count > 0:
                features[b, pos] = 1.0
            for _ in range(quote_count):
                in_quote = not in_quote
    
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

def estimate_loss(model, config, ctx, get_batch_fn, timer):
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

def estimate_probe_loss(model, probe_cluster, config, ctx, get_batch_fn, timer):
    """Estimate loss for all probes"""
    model.eval()
    probe_cluster.set_eval_mode()  # Set probes to eval mode
    out = {'train':{}, 'val':{}}
    
    for split in ['train', 'val']:
        probe_losses = torch.zeros(config.eval_iters, probe_cluster.get_num_probes(), device=config.device)
        probe_accuracies = torch.zeros(config.eval_iters, probe_cluster.get_num_probes(), device=config.device)
        
        for k in range(config.eval_iters):
            X, Y, probe_targets = get_batch_fn(split) 
            with ctx:
                _, _, activations = model(X, Y)
            
            # Get losses for each probe
            timer.start('compute_probe_losses')
            losses = probe_cluster.compute_probe_losses(activations, probe_targets)
            probe_losses[k] = torch.tensor([loss.item() for loss in losses], device=config.device)
            timer.end('compute_probe_losses')
            
            # Get accuracies for each probe
            timer.start('compute_probe_accuracies')
            accs = probe_cluster.compute_accuracies(activations, probe_targets)
            probe_accuracies[k] = torch.tensor(accs, device=config.device)
            timer.end('compute_probe_accuracies')
        
        # Average over evaluation iterations
        out[split]["loss"] = probe_losses.mean(dim=0).tolist()
        out[split]["accuracy"] = probe_accuracies.mean(dim=0).tolist()

    probe_cluster.set_train_mode()  # Set probes back to train mode
    model.train()
    return out

def setup_environment(config, device_type):
    """Set up training environment including device and precision settings."""
    # Enable tensor cores and memory setting
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
    
    return config, ctx, ptdtype

def load_dataset(data_dir):
    """Load training and validation datasets."""
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    return train_data, val_data

def get_model_args(config, data_dir, device):
    """Determine vocabulary size from checkpoint or metadata."""
    if config.init_from == 'huggingface':
        hf_config = AutoConfig.from_pretrained(config.huggingface_model_id)
        for key, value in hf_config.to_dict().items():
            if key not in ['num_layers', 'num_heads', 'hidden_size', 'window_size']:
                setattr(config, key, value)
        if hasattr(hf_config, 'num_layers'):
            config.n_layer = hf_config.num_layers
        if hasattr(hf_config, 'num_heads'):
            config.n_head = hf_config.num_heads
        if hasattr(hf_config, 'hidden_size'):
            config.n_embd = hf_config.hidden_size
        return config, None
    elif config.init_from == 'resume':
        ckpt_path = os.path.join(config.source_dir, 'ckpt.pt')
        print(f"Loading model from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'model_args' in checkpoint:
            model_args_dict = checkpoint['model_args']
            config.vocab_size = model_args_dict.get('vocab_size', None)
        else:
            # Try to infer from weights
            for k in checkpoint['model'].keys():
                if k == 'transformer.wte.weight' or k == '_orig_mod.transformer.wte.weight':
                    config.vocab_size = checkpoint['model'][k].shape[0]
                    print(f"Inferred vocab_size = {config.vocab_size} from checkpoint weights")
                    break
        return config, checkpoint
    else:
        # Find vocab size from meta.pkl
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            import pickle
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            config.vocab_size = meta['vocab_size']
        return config, None

def load_model_from_checkpoint(checkpoint, device, GPT, GPTConfig, return_tokenizer=False):
    """Load model from checkpoint"""
    print(f"Loading model from checkpoint...")
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load model state
    state_dict = checkpoint['model']
    
    # Fix key prefixes if needed (for models saved with DDP)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)

    iter_num = checkpoint.get('iter_num', 0)
    if train_adversarially:
        iter_num = 0
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Get encoder info
    if return_tokenizer and 'config' in checkpoint and checkpoint['config'].get('dataset'):
        data_dir = os.path.join('data', checkpoint['config']['dataset'])
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            encoder_path = os.path.join(data_dir, 'encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoder = pickle.load(f)
    
    # If no encoder available, return None for encoder
    if return_tokenizer:
        return model, gptconf, encoder, iter_num, best_val_loss
    else:
        return model, gptconf, iter_num, best_val_loss

def load_model_from_huggingface(model_id, device, GPT, GPTConfig, return_tokenizer=False):
    """
    Load a model from HuggingFace and convert to our model format.
    """
    print(f"Loading {model_id} from HuggingFace...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
    hf_model.to(device)
    
    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    hf_config = hf_model.config
    model_args = GPTConfig(
        n_layer=hf_config.num_layers,
        n_head=hf_config.num_heads,
        n_embd=hf_config.hidden_size,
        block_size=hf_config.max_position_embeddings,
        bias=True,
        vocab_size=hf_config.vocab_size,
        qkv_bias=False,
        window_size=hf_config.window_size,
        attention_layers=hf_config.attention_layers,
        attn_dropout=hf_config.attention_dropout,
        resid_dropout=hf_config.resid_dropout,
    )
    model = GPT(model_args)
    
    # Load state dict into our model
    model.load_state_dict(hf_model.state_dict())
    # target_block_size = config.block_size
    # model.crop_block_size(target_block_size)
    model.to(device)
    
    if return_tokenizer:
        return model, model_args, tokenizer
    else:
        return model, model_args

def initialize_probes(config, device, checkpoint=None, ProbeCluster=None):
    """Initialize probe cluster if needed."""
    
    probe_cluster = ProbeCluster(
        n_layer=config.n_layer,
        d_model=config.n_embd,
        learning_rate=config.probe_learning_rate,
        device=device
    )

    # Load probe cluster state if available
    if checkpoint is not None and 'probe_state' in checkpoint:
        probe_cluster.load_state_dict(checkpoint['probe_state'])
        
    return probe_cluster

def setup_training_components(model, config, device_type):
    """Set up optimizer, scaler, and other training components."""
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
    
    # Create gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    return model, optimizer, scaler

def create_batch_getter(train_data, val_data, config, device, timer, get_batch_fn):
    """Create a bound get_batch function for easy reuse."""
    def get_batch_bound(split):
        timer.start('get_batch')
        batch = get_batch_fn(split, train_data, val_data, config, device, timer)
        timer.end('get_batch')
        return batch
    return get_batch_bound

def update_learning_rate(optimizer, config, iter_num):
    """Update learning rate according to schedule."""
    if not config.decay_lr:
        return config.learning_rate
        
    # Linear warmup then cosine decay
    if iter_num < config.warmup_iters:
        lr = config.learning_rate * iter_num / config.warmup_iters
    else:
        decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        decay_ratio = min(decay_ratio, 1.0)
        lr = config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1.0 + math.cos(math.pi * decay_ratio))
    
    # Set learning rate with gradient accumulation adjustment
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * config.gradient_accumulation_steps
        
    return lr

def save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, config, probe_cluster=None, final=False):
    """Save model checkpoint."""
    if isinstance(model_args, object) and not isinstance(model_args, dict):
        model_args = {k: v for k, v in vars(model_args).items()}

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }

    if final:
        checkpoint['config'] = vars(config) if hasattr(config, '__dict__') else dict(config)

    if probe_cluster is not None:
        checkpoint['probe_state'] = probe_cluster.state_dict()

    os.makedirs(config.dest_dir, exist_ok=True)
    
    filename = f'ckpt_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt' if final else f'ckpt_intermediate_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt'
    filepath = os.path.join(config.dest_dir, filename)
    print(f"Saving checkpoint to {filepath}")
    torch.save(checkpoint, filepath)

def setup_wandb(config):
    """Initialize wandb logging."""
    if os.environ.get('WANDB_SWEEP_ID'):
        run_name = f"lambda_{config.lambda_adversarial}_phi_{config.phi_probe_steps_per_model_update}_iters_{config.max_iters}"
    else:
        run_name = config.wandb_run_name
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=vars(config)
    )

    wandb.define_metric("iteration", step_metric="iteration")
    wandb.define_metric("train/*", step_metric="iteration")
    wandb.define_metric("val/*", step_metric="iteration")

def log_evaluation_results(eval_loss, probe_losses, iter_num, lr, config):
    """Log evaluation results to console and wandb."""
    print(f"Evaluation results at iteration {iter_num}: train loss {eval_loss['train']:.4f}, val loss {eval_loss['val']:.4f}")

    if not config.wandb_log:
        return
        
    log_dict = {
        "iteration": iter_num,
        "train/model_loss": eval_loss['train'],
        "val/model_loss": eval_loss['val'],
        "train/lr": lr,
    }

    if probe_losses:
        print("Probe losses and accuracies:")
        for split in ['train', 'val']:
            for stat in probe_losses[split]:
                for i, value in enumerate(probe_losses[split][stat]):
                    probe_type = 'attn' if i % 2 == 0 else 'MLP'
                    layer = i // 2
                    print(f"{split} probe {i} ({probe_type}-{layer}) {stat}: {value:.4f}")
            
                    if config.wandb_log:
                        log_dict[f"{split}/probe-{stat}-{probe_type}-{layer}"] = value
    
    if config.wandb_log:
        wandb.log(log_dict)

def debug_print_batch(X, Y, probe_targets):
    """Print sample batch data for debugging."""        
    enc = tiktoken.get_encoding("gpt2")
    print("\n=== Sample Batch Data ===")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, probe_targets shape: {probe_targets.shape}")
    print(f"\033[1mX sample:\033[0m\n{repr(enc.decode(X[0, :-1].tolist()))}")
    print(f"\033[1mY sample:\033[0m\n{repr(enc.decode(Y[0, :-1].tolist()))}")
    print(f"\033[1mprobe_targets sample:\033[0m\n{[(enc.decode([X[0, i].item()]), probe_targets[0, i].item()) for i in range(len(X[0, :-1]))]}".replace("), ", "),\n"))
    print("=========================\n")

def run_single_training_step(model, probe_cluster, optimizer, scaler, 
                           X, Y, probe_targets, config, ctx, iter_num, get_batch_fn):
    """Execute a single training step."""
    # Zero gradients if first iteration
    if iter_num == 0:
        optimizer.zero_grad(set_to_none=True)
        if config.train_probes:
            probe_cluster.zero_grad_probes()
    
    # Forward pass
    with ctx:
        logits, transformer_loss, activations = model(X, Y, use_checkpoint=config.use_gradient_checkpointing)
        
        # Process probes and adversarial training
        if config.train_probes and config.train_adversarially:
            # Compute probe losses directly
            probe_losses = probe_cluster.compute_probe_losses(activations, probe_targets)
            adversarial_loss = -sum(probe_losses) * config.lambda_adversarial
            total_loss = transformer_loss + adversarial_loss
        else:
            total_loss = transformer_loss
    
    # Backward pass
    if config.train_model:
        scaler.scale(total_loss).backward()

        if (iter_num + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            if config.train_probes:
                probe_cluster.zero_grad_probes()
    
    # Update probes based on phi_probe_steps_per_model_update
    if config.train_probes and config.train_adversarially:
        update_probes(model, probe_cluster, activations, probe_targets, 
                      config, scaler, get_batch_fn)
                
    return transformer_loss.item()

def update_probes(model, probe_cluster, activations, probe_targets, 
                 config, scaler, get_batch_fn):
    """Update probes with current activations and possibly new batches."""
    # Simple probe update with detached activations
    with torch.no_grad():
        detached_activations = [act.detach() for act in activations]
    
    # Update probes multiple times per model update
    for i in range(config.phi_probe_steps_per_model_update):
        # Update probes with current batch
        probe_cluster.update_probes(
            detached_activations,
            probe_targets,
            scaler=scaler if config.dtype == 'float16' else None
        )
        
        # Get a new batch for next probe update if not the last one
        if i < config.phi_probe_steps_per_model_update - 1:
            X_new, Y_new, probe_targets_new = get_batch_fn('train')
            with torch.no_grad():
                # Get new activations without recomputing gradients
                _, _, new_activations = model(X_new, Y_new, use_checkpoint=config.use_gradient_checkpointing)
                detached_activations = [act.detach() for act in new_activations]
            probe_targets = probe_targets_new

def run_evaluation(model, probe_cluster, config, ctx, get_batch_fn, timer):
    """Run model and probe evaluation."""
    timer.start('evaluation')
    eval_loss = estimate_loss(model, config, ctx, get_batch_fn, timer)
    timer.end('evaluation')
    
    probe_losses = None
    if config.train_probes:
        timer.start('probe_evaluation')
        probe_losses = estimate_probe_loss(model, probe_cluster, config, ctx, get_batch_fn, timer)
        timer.end('probe_evaluation')
    
    return eval_loss, probe_losses

def setup_signal_handler():
    """Create and return a signal handler for graceful interruption."""
    stop_requested = [False]  # Use list for mutable state
    
    def signal_handler(sig, frame):
        stop_requested[0] = True
        
    return stop_requested, signal_handler