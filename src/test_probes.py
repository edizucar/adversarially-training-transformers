import json
import torch
import argparse
from colorama import Fore, Style

from probes import ProbeCluster
from model import GPTConfig, GPT
from utils import load_model_from_checkpoint, load_model_from_huggingface, load_probes_from_checkpoint
from inference_utils import setup_pytorch, encode_prompt, decode_tokens, generate, format_probe_scores, setup_matplotlib, plot_probe_heatmap

def parse_args():
    parser = argparse.ArgumentParser(description='Test probes')
    parser.add_argument('--probe_checkpoint', type=str, default="../checkpoints/tiny_stories_adv/latest.pt", help='Path to probe checkpoint file')
    parser.add_argument('--probe_stride', type=int, default=5, help='Computes probe scores one out of every N tokens')
    parser.add_argument('--probe_plot', action='store_true', help='Generate a heatmap visualization of probe scores')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Path to model checkpoint file')
    parser.add_argument('--model_huggingface', type=str, default="roneneldan/TinyStories-33M", help='HuggingFace model ID')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    return parser.parse_args()

def main():
    args = parse_args()

    with open('../prompts/probe_grader.json', 'r') as f:
        prompts = json.load(f)
    prompt_text = prompts[0]['text']

    device_type = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    device = torch.device(device_type)
    ctx, _ = setup_pytorch(args.seed, device_type)
    
    # Load model and encoder
    if args.model_checkpoint:
        model, model_args, encoder, _, _ = load_model_from_checkpoint(args.model_checkpoint, device, GPT, GPTConfig, return_tokenizer=True)
    else:
        model, model_args, encoder = load_model_from_huggingface(args.model_huggingface, device, GPT, GPTConfig, return_tokenizer=True)
        
    probe_cluster = load_probes_from_checkpoint(args.probe_checkpoint, device, ProbeCluster)
    assert probe_cluster is not None, "No probes found or failed to load probes"

    probe_cluster.set_eval_mode()
    print(f"Loaded {probe_cluster.get_num_probes()} probes from {args.probe_checkpoint}")
    
    prompt_tokens = encode_prompt(prompt_text, encoder, device)
    
    print(f"\nGenerating with prompt: {prompt_text}")
    print("-" * 40)
    
    generated_tokens, probe_scores = generate(
        model,
        prompt_tokens,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        device,
        ctx,
        probe_cluster,
        args.probe_stride
    )
    tokens = []
    for token in generated_tokens:
        token_text = decode_tokens([token], encoder)
        tokens.append(token_text)
    
    print(prompt_text, end="")
    for i, (token, scores) in enumerate(zip(tokens, probe_scores)):
        if scores is not None: # Print scores every args.probe_stride tokens
            print(f"{Fore.YELLOW}{Style.BRIGHT}{token}{Style.RESET_ALL}", end="")
            formatted_scores = format_probe_scores(scores, probe_cluster.n_layer)
            print(f"\n  -> Probe scores: {formatted_scores}\n", end="")
        else:
            print(f"{token}", end="")

    if args.probe_plot:
        setup_matplotlib()
        plot_probe_heatmap(prompt_text, tokens, probe_scores, save_path='probe_plot.png')

    generated_text = decode_tokens(generated_tokens, encoder)
    print(f"\n\nFull text\n---\n{prompt_text}{generated_text}")

if __name__ == "__main__":
    main()