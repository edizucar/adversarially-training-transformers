#!/usr/bin/env python
"""
Inference script for GPT model checkpoint.
"""

import os
import argparse
import pickle
import numpy as np
import torch
from contextlib import nullcontext

from model import GPTConfig, GPT
from config import TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with a GPT model checkpoint')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/tiny_stories/ckpt.pt", required=True, help='Path to checkpoint file')
    parser.add_argument('--prompt', type=str, default="Once upon", help='Text prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    return parser.parse_args()

def setup_pytorch(seed, device_type):
    """Set up PyTorch settings"""
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup dtype and autocast context
    ptdtype = torch.float16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return ctx, ptdtype

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model arguments
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
    model.eval()
    
    # Get encoder info
    if 'config' in checkpoint and checkpoint['config'].get('dataset'):
        data_dir = os.path.join('data', checkpoint['config']['dataset'])
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            # Load encoder if available
            encoder_path = os.path.join(data_dir, 'encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoder = pickle.load(f)
                return model, encoder
    
    # If no encoder available, return None for encoder
    return model, None

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
def generate(model, prompt_tokens, max_new_tokens, temperature, top_k, top_p, device, ctx):
    """Generate text using the model"""
    # Start with prompt tokens
    x = prompt_tokens.to(device)
    generation = []
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        with ctx:
            logits, _, _ = model(x)
        
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
        
        # Optionally break if an end token is generated
        # This depends on your tokenizer, for some cases you might want to check for EOS token
    
    return generation

def main():
    args = parse_args()
    
    # Determine device
    device_type = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    device = torch.device(device_type)
    
    # Setup PyTorch context
    ctx, _ = setup_pytorch(args.seed, device_type)
    
    # Load model and encoder
    model, encoder = load_model(args.checkpoint, device)
    
    # Process the prompt
    prompt_tokens = encode_prompt(args.prompt, encoder, device)
    
    print(f"\nGenerating with prompt: {args.prompt}")
    print("-" * 40)
    
    # Generate text
    generated_tokens = generate(
        model, 
        prompt_tokens, 
        args.max_new_tokens, 
        args.temperature, 
        args.top_k, 
        args.top_p, 
        device, 
        ctx
    )
    
    # Convert tokens to text
    prompt_text = decode_tokens(prompt_tokens[0].tolist(), encoder) if args.prompt else ""
    generated_text = decode_tokens(generated_tokens, encoder)
    
    # Print the result
    print(f"{prompt_text}{generated_text}")

if __name__ == "__main__":
    main()