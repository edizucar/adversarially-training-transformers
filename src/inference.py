#!/usr/bin/env python
"""
Inference script for GPT model checkpoint.
"""

import os
import argparse
import pickle
import numpy as np
import torch
import math
from colorama import Fore, Style
from contextlib import nullcontext
from itertools import zip_longest
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes import ProbeCluster
from model import GPTConfig, GPT
from config import TrainingConfig
from utils import load_model_from_checkpoint, load_model_from_huggingface

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with a GPT model checkpoint')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/tiny_stories/ckpt.pt", help='Path to checkpoint file')
    parser.add_argument('--huggingface', type=str, default=None, help='HuggingFace model ID')
    parser.add_argument('--prompt', type=str, default="Once upon a time there was", help='Text prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    parser.add_argument('--run_probes', action='store_true', help='Whether to run probe inference on generated tokens')
    parser.add_argument('--probe_checkpoint', type=str, default=None, help='Optional separate checkpoint for probes')
    parser.add_argument('--probe_stride', type=int, default=5, help='Computes probe scores one out of every N tokens')
    parser.add_argument('--probe_plot', action='store_true', help='Generate a heatmap visualization of probe scores')
    return parser.parse_args()

def setup_pytorch(seed, device_type):
    """Set up PyTorch settings"""
    if seed is None:
        seed = np.random.randint(0, 1000000)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup dtype and autocast context
    ptdtype = torch.float16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return ctx, ptdtype

def encode_prompt(prompt, encoder, device):
    """Encode the prompt to token IDs"""
    if not prompt:
        return torch.zeros((1, 1), dtype=torch.long, device=device)
    
    if encoder:
        # If we have an encoder, use it
        tokens = encoder.encode(prompt)
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        token_tensor = token_tensor.unsqueeze(0)  # Add batch dimension
    else:
        # Simple fallback for open-source models without encoder
        # This assumes GPT-2 tokenization for simplicity
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokens = tokenizer.encode(prompt)
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            token_tensor = token_tensor.unsqueeze(0)  # Add batch dimension
        except:
            print("No encoder found and couldn't load GPT2Tokenizer. Using empty prompt.")
            return torch.zeros((1, 1), dtype=torch.long, device=device)
    
    return token_tensor

def decode_tokens(tokens, encoder):
    """Convert token IDs back to text"""
    if encoder and hasattr(encoder, 'decode'):
        return encoder.decode(tokens)
    
    # Fallback for open-source models
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return tokenizer.decode(tokens)
    except:
        return f"[Unable to decode tokens: {tokens}]"

@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens, temperature, top_k, top_p, device, ctx, probe_cluster=None, probe_stride=1):
    """Generate text using the model and score with probes"""
    # Start with prompt tokens
    x = prompt_tokens.to(device)
    generation = []
    probe_scores = []
    
    # Generate tokens one by one
    for i in range(max_new_tokens):
        with ctx:
            logits, _, activations = model(x)
        
        # Get logits for the last token
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Apply top-p filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('Inf')
        
        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Add the token to the generation and to the context
        generation.append(next_token.item())
        x = torch.cat((x, next_token), dim=1)
        
        # Score with probes if we have them
        if probe_cluster is not None:
            if i % probe_stride == 0:
                # Extract the last token's activations
                last_activations = [act[:, -1:, :].detach() for act in activations]
                
                # Get probe predictions (probabilities)
                with torch.no_grad():
                    logits = [probe(act).squeeze(-1) for act, probe in zip(last_activations, probe_cluster.probes)]
                    probs = [torch.sigmoid(l) for l in logits]
                    scores = [p.item() for p in probs]
                
                probe_scores.append(scores)
            else:
                probe_scores.append(None)
    
    return generation, probe_scores

def load_probes_from_checkpoint(checkpoint_path, device, ProbeCluster=None):
    """Load probe cluster from checkpoint"""
    from probes import ProbeCluster as DefaultProbeCluster
    
    if ProbeCluster is None:
        ProbeCluster = DefaultProbeCluster
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'probe_state' in checkpoint:
        # New format
        n_layer = checkpoint['model_args'].get('n_layer', 12)
        d_model = checkpoint['model_args'].get('n_embd', 768)
        
        probe_cluster = ProbeCluster(
            n_layer=n_layer,
            d_model=d_model,
            learning_rate=1e-3,  # Default value, not used for inference
            device=device
        )
        probe_cluster.load_state_dicts(checkpoint['probe_state']['probe_state_dicts'])
        return probe_cluster
    
    # Try older format compatibility
    elif any(key.startswith('probe_') for key in checkpoint.keys()):
        # Old format from ProbeIntervention
        try:
            from old_files.ProbeIntervention import ProbeCluster as OldProbeCluster
            return OldProbeCluster.load_from_checkpoint(checkpoint, lr=1e-3, device=device)
        except (ImportError, KeyError) as e:
            print(f"Failed to load probes: {e}")
            return None
    
    print("No probes found in checkpoint")
    return None

def format_probe_scores(scores, n_layer):
    """Format probe scores in a readable way
    
    Args:
        scores: List of scores for each probe
        n_layer: Number of transformer layers (used to label probes)
        
    Returns:
        String with formatted scores
    """
    result = []
    for i, score in enumerate(scores):
        probe_type = 'attn' if i % 2 == 0 else 'MLP'
        layer = i // 2
        result.append(f"{probe_type}-{layer}: {score:.2f}")
    
    return " | ".join(result)

def plot_probe_heatmap(prompt_text, tokens, probe_scores, save_path='probe_plot.png'):
    """
    Generate a heatmap visualization of probe scores.
    
    Args:
        prompt_text: Text of the prompt
        tokens: List of token strings
        probe_scores: List of probe scores for each token (None if not computed)
        save_path: Path to save the plot
    """
    # Filter out tokens without probe scores
    valid_indices = [i for i, scores in enumerate(probe_scores) if scores is not None]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_scores = [probe_scores[i] for i in valid_indices]
    
    if not valid_scores:
        print("No probe scores to plot")
        return
    
    # Convert scores to numpy array
    score_array = np.array(valid_scores)
    
    # Determine if each token is in a quote
    full_text = prompt_text + ''.join(tokens)
    in_quote = []
    quote_status = False
    
    for i in valid_indices:
        # Get position of this token in the full text
        pos = len(prompt_text) + len(''.join(tokens[:i]))
        
        # Check if we've passed an odd number of quotes before this position
        quote_count = full_text[:pos].count('"')
        current_quote_status = quote_count % 2 == 1
        
        in_quote.append(current_quote_status)
    
    # Create labels for the y-axis (tokens with quote indicators)
    y_labels = []
    for token, quoted in zip(valid_tokens, in_quote):
        # Clean and truncate token for display
        token_clean = token.replace('\n', '\\n').strip()
        if len(token_clean) > 15:
            token_clean = token_clean[:12] + '...'
        
        # No quote indicator in the label text anymore
        y_labels.append(token_clean)
    
    # Create labels for the x-axis (probe names)
    num_probes = len(valid_scores[0])
    x_labels = []
    for i in range(num_probes):
        probe_type = 'attn' if i % 2 == 0 else 'MLP'
        layer = i // 2
        x_labels.append(f"{probe_type}-{layer}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(valid_tokens) * 0.3)))
    
    # Create a custom white-to-red colormap
    colors = [(1, 1, 1), (1, 0.7, 0.5), (1, 0.4, 0.2), (1, 0, 0)]  # white -> light orange -> orange -> red
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('WhiteToRed', colors)
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Assuming scores are between 0 and 1
    
    # Fix reversed order by reversing the score array
    score_array = np.flipud(score_array)
    y_labels = y_labels[::-1]  # Also reverse the labels
    in_quote = in_quote[::-1]  # And the quote indicators
    
    # Create the heatmap with no spacing
    heatmap = ax.pcolormesh(score_array, cmap=custom_cmap, norm=norm, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Probe Score')
    
    # Set axis labels and ticks
    ax.xaxis.tick_top()  # Move x-axis ticks to top
    ax.xaxis.set_label_position('top')  # Move x-axis label to top
    
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(num_probes) + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha='left')
    
    # Highlight quoted tokens in the axis labels
    for i, quoted in enumerate(in_quote):
        if quoted:
            text = ax.get_yticklabels()[i]
            text.set_color('red')
            text.set_fontweight('bold')
    
    # Title at the top
    plt.title('Probe Scores per Token', pad=20)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    print(f"\n\nSaved probe visualization to {save_path}")

def main():
    args = parse_args()
    
    # Determine device
    device_type = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    device = torch.device(device_type)
    
    # Setup PyTorch context
    ctx, _ = setup_pytorch(args.seed, device_type)
    
    # Load model and encoder
    probe_cluster = None
    if args.huggingface:
        model, model_args, encoder = load_model_from_huggingface(args.huggingface, device, GPT, GPTConfig, return_tokenizer=True)
        probe_path = args.probe_checkpoint if (args.run_probes and args.probe_checkpoint) else None
    else:
        model, model_args, encoder, _, _ = load_model_from_checkpoint(args.checkpoint, device, GPT, GPTConfig, return_tokenizer=True)
        probe_path = args.probe_checkpoint if args.probe_checkpoint else args.checkpoint
        
    if args.run_probes and probe_path:
        probe_cluster = load_probes_from_checkpoint(probe_path, device, ProbeCluster)
        if probe_cluster:
            probe_cluster.set_eval_mode()
            print(f"Loaded {probe_cluster.get_num_probes()} probes from {probe_path}")
        else:
            print("No probes found or failed to load probes")
    
    # Process the prompt
    prompt_tokens = encode_prompt(args.prompt, encoder, device)
    
    print(f"\nGenerating with prompt: {args.prompt}")
    print("-" * 40)
    
    generated_tokens, probe_scores = generate(
        model,
        prompt_tokens,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        device,
        ctx,
        probe_cluster,
        args.probe_stride
    )
    
    # Convert tokens to text
    prompt_text = decode_tokens(prompt_tokens[0].tolist(), encoder) if args.prompt else ""
    
    # Display generated text with probe scores if available
    if probe_scores:
        # Decode tokens individually for alignment with scores
        tokens = []
        for token in generated_tokens:
            token_text = decode_tokens([token], encoder)
            tokens.append(token_text)
        
        # Print prompt first
        print(prompt_text, end="")
        # Print each token with its probe scores
        for i, (token, scores) in enumerate(zip(tokens, probe_scores)):
            if scores is not None: # Print scores every 5 tokens
                print(f"{Fore.YELLOW}{Style.BRIGHT}{token}{Style.RESET_ALL}", end="")
                formatted_scores = format_probe_scores(scores, probe_cluster.n_layer)
                print(f"\n  -> Probe scores: {formatted_scores}\n", end="")
            else:
                print(f"{token}", end="")
        
        # Create a visualization of probe scores if requested
        if args.probe_plot:
            plot_probe_heatmap(prompt_text, tokens, probe_scores, save_path='probe_plot.png')
    else:
        # Just print the generated text without probe scores
        generated_text = decode_tokens(generated_tokens, encoder)
        print(f"{prompt_text}{generated_text}")

if __name__ == "__main__":
    main()