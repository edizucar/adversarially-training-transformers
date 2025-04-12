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
import matplotlib.font_manager as fm

from probes import ProbeCluster
from model import GPTConfig, GPT

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
        # Fallback for open-source models without encoder
        # This assumes GPT-2 tokenization for simplicity
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokens = tokenizer.encode(prompt)
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            token_tensor = token_tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"No encoder found and couldn't load GPT2Tokenizer. Using empty prompt. Error: {e}")
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
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        return f"[Unable to decode tokens: {tokens}]"

@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens, temperature, top_k, top_p, device, ctx, probe_cluster=None, probe_stride=1):
    """
    Generate text using the model and score with probes
    
    Args:
        model: Model to use for generation
        prompt_tokens: Tokenized prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Top-k filtering
        top_p: Top-p filtering
        device: Device to use for generation
        ctx: Context for automatic mixed precision
        probe_cluster: Probe cluster to use for scoring
        probe_stride: Scores one in every probe_stride tokens
    """
    # Start with prompt tokens
    x = prompt_tokens.to(device)
    generation = []
    probe_scores = []
    
    # Generate tokens one by one
    for i in range(max_new_tokens):
        with ctx:
            logits, _, activations = model(x)
        
        next_token = sample_from_logits(logits[:, -1, :], temperature, top_k, top_p)
        generation.append(next_token.item())
        x = torch.cat((x, next_token), dim=1)
        
        # Score with probes if we have them
        if probe_cluster is not None and i % probe_stride == 0:
            scores = score_with_probes(activations, probe_cluster)
            probe_scores.append(scores)
        else:
            probe_scores.append(None)
    
    return generation, probe_scores

def sample_from_logits(logits, temperature, top_k, top_p):
    """Sample next token from logits"""
    logits = logits / temperature
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float('Inf')
    elif top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def score_with_probes(activations, probe_cluster):
    """Score with probes"""
    last_activations = [act[:, -1:, :].detach() for act in activations]
    with torch.no_grad():
        logits = [probe(act).squeeze(-1) for act, probe in zip(last_activations, probe_cluster.probes)]
        probs = [torch.sigmoid(l) for l in logits]
        scores = [p.item() for p in probs]
    return scores

def format_probe_scores(scores, n_layer):
    """Format probe scores in a readable way for printing
    
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

def setup_matplotlib():
    """Set up matplotlib with custom fonts and style."""
    fm.fontManager.addfont('../fonts/Montserrat-Regular.ttf')
    plt.rcParams['font.family'] = 'Montserrat'
    plt.rcParams['hatch.linewidth'] = 0.3  # Thinner hatch lines

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