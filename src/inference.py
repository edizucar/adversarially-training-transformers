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
from contextlib import nullcontext
from itertools import zip_longest
# from model import GPTConfig, GPT
from model_test import GPTConfig, GPT
from config import TrainingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with a GPT model checkpoint')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/tiny_stories/ckpt.pt", help='Path to checkpoint file')
    parser.add_argument('--huggingface', type=str, default="roneneldan/TinyStories-33M", help='HuggingFace model ID')
    parser.add_argument('--prompt', type=str, default="Once upon a time there was", help='Text prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
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

def load_model_from_checkpoint(checkpoint_path, device):
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

def load_model_from_huggingface(model_id, device):
    """
    Load the TinyStories-33M model from HuggingFace and convert to our model format.
    """
    print("Loading TinyStories-33M from HuggingFace...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
    hf_model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    hf_config = hf_model.config
    model_args = GPTConfig(
        n_layer=hf_config.num_layers,
        n_head=hf_config.num_heads,
        n_embd=hf_config.hidden_size,
        block_size=hf_config.max_position_embeddings,
        bias=True,
        vocab_size=hf_config.vocab_size,
    )
    model_args.qkv_bias = False
    model_args.window_size = hf_config.window_size
    model_args.attn_dropout = hf_config.attention_dropout
    model_args.resid_dropout = hf_config.resid_dropout
    model_args.attention_layers = hf_config.attention_layers
    
    model = GPT(model_args)
    
    # Load state dict into our model
    model.load_state_dict(hf_model.state_dict())
    model.to(device)

    test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    with torch.no_grad():
        print("\n===== QKV COMPARISON =====")
        compare_qkv(model, hf_model, test_input, layer_idx=0)
        print("\n===== FULL FORWARD PASS COMPARISON =====")
        debug_forward(model, hf_model, test_input)
    
    return model, tokenizer

def debug_forward(my_model, hf_model, input_ids):
    """
    Debug forward pass comparison between HF model and model it's state_dict is loaded into
    """
    # Get embeddings
    my_emb = my_model.transformer.wte(input_ids) + my_model.transformer.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
    hf_emb = hf_model.transformer.wte(input_ids) + hf_model.transformer.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
    
    print(f"Embedding diff: {(my_emb - hf_emb).abs().max().item()}")
    
    # Track layer outputs
    my_x = my_emb
    hf_x = hf_emb
    
    for i in range(len(my_model.transformer.h)):
        # Get layer outputs from your model
        my_attn_out = my_model.transformer.h[i].attn(my_model.transformer.h[i].ln_1(my_x))
        my_x_temp = my_x + my_attn_out
        my_x_next = my_x_temp + my_model.transformer.h[i].mlp(my_model.transformer.h[i].ln_2(my_x_temp))
        
        # Get layer outputs from HF model
        hf_attn_out = hf_model.transformer.h[i].attn(hf_model.transformer.h[i].ln_1(hf_x))
        hf_x_temp = hf_x + hf_attn_out[0]
        hf_x_next = hf_x_temp + hf_model.transformer.h[i].mlp(hf_model.transformer.h[i].ln_2(hf_x_temp))

        # Test attention output on each other's mlp
        my_attn_out_mlp = my_model.transformer.h[i].mlp(my_model.transformer.h[i].ln_2(hf_x_temp))
        hf_attn_out_mlp = hf_model.transformer.h[i].mlp(hf_model.transformer.h[i].ln_2(hf_x_temp))
        
        print(f"Layer {i} attention diff: {(my_attn_out - hf_attn_out[0]).abs().max().item()}")
        print(f"Layer {i} output diff: {(my_x_next - hf_x_next).abs().max().item()}")
        print(f"Layer {i} mlp diff: {(my_attn_out_mlp - hf_attn_out_mlp).abs().max().item()}")
        
        my_x = my_x_next
        hf_x = hf_x_next
    
    # Final layer norm
    my_final = my_model.transformer.ln_f(my_x)
    hf_final = hf_model.transformer.ln_f(hf_x)
    print(f"Final LN diff: {(my_final - hf_final).abs().max().item()}")
    
    # Logits
    my_logits = my_model.lm_head(my_final)
    hf_logits = hf_model.lm_head(hf_final)
    print(f"Logit diff: {(my_logits - hf_logits).abs().max().item()}")

def compare_qkv(my_model, hf_model, input_ids, layer_idx=0):
    """
    Compare QKV projections between HF model and model it's state_dict is loaded into
    """
    # Get embeddings
    my_emb = my_model.transformer.wte(input_ids) + my_model.transformer.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
    hf_emb = hf_model.transformer.wte(input_ids) + hf_model.transformer.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
    
    # Get layer norm output
    my_ln1 = my_model.transformer.h[layer_idx].ln_1(my_emb)
    hf_ln1 = hf_model.transformer.h[layer_idx].ln_1(hf_emb)
    
    # QKV projections
    my_q = my_model.transformer.h[layer_idx].attn.attention.q_proj(my_ln1)
    my_k = my_model.transformer.h[layer_idx].attn.attention.k_proj(my_ln1)
    my_v = my_model.transformer.h[layer_idx].attn.attention.v_proj(my_ln1)
    
    hf_q = hf_model.transformer.h[layer_idx].attn.attention.q_proj(hf_ln1)
    hf_k = hf_model.transformer.h[layer_idx].attn.attention.k_proj(hf_ln1)
    hf_v = hf_model.transformer.h[layer_idx].attn.attention.v_proj(hf_ln1)
    
    print(f"Q diff: {(my_q - hf_q).abs().max().item()}")
    print(f"K diff: {(my_k - hf_k).abs().max().item()}")
    print(f"V diff: {(my_v - hf_v).abs().max().item()}")
    
    # Check attention product
    B, T, C = my_ln1.size()
    num_heads = my_model.transformer.h[layer_idx].attn.attention.n_head
    head_size = C // num_heads
    
    my_q = my_q.view(B, T, num_heads, head_size).transpose(1, 2)
    my_k = my_k.view(B, T, num_heads, head_size).transpose(1, 2)
    my_v = my_v.view(B, T, num_heads, head_size).transpose(1, 2)
    
    hf_q = hf_q.view(B, T, num_heads, head_size).transpose(1, 2)
    hf_k = hf_k.view(B, T, num_heads, head_size).transpose(1, 2)
    hf_v = hf_v.view(B, T, num_heads, head_size).transpose(1, 2)
    
    my_attn = (my_q @ my_k.transpose(-2, -1)) * (1.0 / math.sqrt(my_k.size(-1)))
    hf_attn = (hf_q @ hf_k.transpose(-2, -1)) * (1.0 / math.sqrt(hf_k.size(-1)))
    
    print(f"Attention score diff: {(my_attn - hf_attn).abs().max().item()}")

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
    if args.huggingface:
        model, encoder = load_model_from_huggingface(args.huggingface, device)
    else:
        model, encoder = load_model_from_checkpoint(args.checkpoint, device)
    
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