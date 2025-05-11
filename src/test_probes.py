import os
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
    parser.add_argument('--all_probes', action='store_true', help='Test all probes')
    parser.add_argument('--probe_stride', type=int, default=5, help='Computes probe scores one out of every N tokens')
    parser.add_argument('--probe_plot', action='store_true', help='Generate a heatmap visualization of probe scores')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Path to model checkpoint file')
    parser.add_argument('--model_huggingface', type=str, default="roneneldan/TinyStories-33M", help='HuggingFace model ID')
    parser.add_argument('--test_clean_and_adversarial', action='store_true', help='Test probe on model adversarially trained against, and clean model')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    return parser.parse_args()

def generate_and_plot(model, prompt_tokens, args, config, iter_num, device, ctx, probe_cluster, encoder, prompt_text, suffix=''):
    while True:
        generated_tokens, probe_scores = generate(
            model, prompt_tokens, args.max_tokens, args.temperature,
            args.top_k, args.top_p, device, ctx, probe_cluster, args.probe_stride
        )
        tokens = [decode_tokens([token], encoder) for token in generated_tokens]
        if any(token == '"' for token in tokens) or any(token == "'" for token in tokens):
            break
        print(f"Rejected generated tokens: {tokens}")

    lambda_adversarial = config['lambda_adversarial']
    phi_probe_steps_per_model_update = config['phi_probe_steps_per_model_update']
    num_iters = iter_num
    
    print(f"\nGenerating with {suffix} for lambda={lambda_adversarial}, phi={phi_probe_steps_per_model_update}, num_iters={num_iters}:")
    print("-" * 40)
    print(prompt_text, end="")
    for i, (token, scores) in enumerate(zip(tokens, probe_scores)):
        if scores is not None:
            print(f"{Fore.YELLOW}{Style.BRIGHT}{token}{Style.RESET_ALL}", end="")
            print(f"\n  -> Probe scores: {format_probe_scores(scores, probe_cluster.n_layer)}\n", end="")
        else:
            print(f"{token}", end="")

    if args.probe_plot:
        plot_probe_heatmap(prompt_text, tokens, probe_scores, save_path=f'scores/probe_l{lambda_adversarial}_p{phi_probe_steps_per_model_update}_i{num_iters}_{suffix}.png')

    return decode_tokens(generated_tokens, encoder)

def main():
    args = parse_args()
    
    with open('../prompts/probe_grader.json', 'r') as f:
        prompt_text = json.load(f)[0]['text']

    device_type = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    device = torch.device(device_type)
    ctx, _ = setup_pytorch(args.seed, device_type)

    if args.all_probes:
        probe_paths = [os.path.join("../checkpoints/tiny_stories_adv", f) for f in os.listdir("../checkpoints/tiny_stories_adv") if f.endswith(".pt") and f != "latest.pt"]
    else:
        probe_paths = [args.probe_checkpoint]
    
    for probe_path in probe_paths:
        # Load probes
        probe_cluster, config, iter_num = load_probes_from_checkpoint(probe_path, device, ProbeCluster)
        assert probe_cluster is not None
        probe_cluster.set_eval_mode()
        print(f"Loaded {probe_cluster.get_num_probes()} probes")
    
        # Load model from checkpoint and/or huggingface
        if args.model_checkpoint:
            checkpoint_model, model_args, _, encoder, _, _ = load_model_from_checkpoint(
                args.model_checkpoint, device, GPT, GPTConfig, return_tokenizer=True)
        else:
            checkpoint_model, model_args, _, encoder, _, _ = load_model_from_checkpoint(
                args.probe_checkpoint, device, GPT, GPTConfig, return_tokenizer=True)
            
        if args.test_clean_and_adversarial or not checkpoint_model:
            hf_model, model_args, encoder = load_model_from_huggingface(
                args.model_huggingface, device, GPT, GPTConfig, return_tokenizer=True)
        
        prompt_tokens = encode_prompt(prompt_text, encoder, device)
        
        if args.probe_plot:
            setup_matplotlib()
        
        if args.test_clean_and_adversarial:
            if checkpoint_model:
                generate_and_plot(checkpoint_model, prompt_tokens, args, config, iter_num, device, ctx, 
                                probe_cluster, encoder, prompt_text, suffix='checkpoint')
            generate_and_plot(hf_model, prompt_tokens, args, config, iter_num, device, ctx,
                            probe_cluster, encoder, prompt_text, suffix='huggingface')
        else:
            model = checkpoint_model if checkpoint_model else hf_model
            generate_and_plot(model, prompt_tokens, args, config, iter_num, device, ctx,
                            probe_cluster, encoder, prompt_text)

if __name__ == "__main__":
    main()