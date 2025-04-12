#!/usr/bin/env python
"""
Inference script for GPT model checkpoint.
"""

import argparse
import torch
from colorama import Fore, Style

from probes import ProbeCluster
from model import GPTConfig, GPT
from utils import load_model_from_checkpoint, load_model_from_huggingface, load_probes_from_checkpoint
from inference_utils import setup_pytorch, encode_prompt, decode_tokens, generate, format_probe_scores, plot_probe_heatmap

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with a GPT model checkpoint')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/tiny_stories_adv/latest.pt", help='Path to checkpoint file')
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
        
        print("\n\nFull text\n---")
    
    generated_text = decode_tokens(generated_tokens, encoder)
    print(f"{prompt_text}{generated_text}")

if __name__ == "__main__":
    main()