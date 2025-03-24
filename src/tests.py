import os
import re
import math
import json
import torch
import random
import tiktoken
import argparse
import colorama
from colorama import Fore, Style
from datasets import load_dataset
from pathlib import Path
from time import perf_counter
from tqdm import tqdm

from utils import odd_quotes_in_text

def find_odd_quotes_in_dataset(dataset="roneneldan/TinyStories", text_field="text", full_text=False):
    """Find examples with an odd number of quotation marks."""
    dataset = load_dataset(dataset)
    odd_quotes = []
    
    for split in ['train', 'validation']:
        for i, example in enumerate(tqdm(dataset[split], desc=f"Processing {split} set")):
            # Handle both dict-style and direct text access
            text = example[text_field] if isinstance(example, dict) else example
            if odd_quotes_in_text(text):
                odd_quotes.append((i, text))
    
    print(f"Found {len(odd_quotes)} examples out of {len(dataset['train']) + len(dataset['validation'])} with an odd number of quotation marks")
    for idx, text in odd_quotes[:3]:
        print(f"\n\n\033[1mExample {idx}:\033[0m\n")
        print(text if full_text else text[:100] + "..." if len(text) > 100 else text)
    
    return odd_quotes

def visualize_quotes(text, tokenizer, quote_detector_fn):
    """Process a text sample and visualize the quote detection results."""
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    result = quote_detector_fn(tokens_tensor, tokenizer)
    features = result.features[0].cpu().numpy()
    token_texts = [tokenizer.decode_single_token_bytes(token).decode('utf-8', errors='replace') for token in tokens]
    
    print(f"\033[1mOriginal text:\033[0m\n{text}\n")
    
    print("\033[1mQuote detection result (yellow = inside quotes):\033[0m")
    for token, is_in_quote in zip(token_texts, features):
        print(f"{Fore.YELLOW}{Style.BRIGHT}{token}{Style.RESET_ALL}" if is_in_quote else token, end="")
    
    in_quote_tokens = [token for token, is_in_quote in zip(token_texts, features) if is_in_quote]
    print(f"\n\n\033[1mIn-quote tokens:\033[0m\n{in_quote_tokens}\n")
    
    quoted_text = re.findall(r'"([^"]*)"', text)
    if quoted_text:
        print("\033[1mText within quotes (found by regex):\033[0m")
        for i, quote in enumerate(quoted_text):
            print(f"{i+1}. \"{quote}\"")
    else:
        print("\033[1mNo quoted text found by regex.\033[0m")
    
    print("-" * 80)

def test_quote_detection(quote_detector_fn, num_samples=10):
    """Test the quote detection function on text samples."""
    colorama.init()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Test simple examples
    simple_examples = [
        'He said "hello" to her.',
        'She replied, "I am fine, thank you."',
        'No quotes in this sentence.',
        '"This entire sentence is in quotes."',
        'Multiple "separate" quotes "in one" sentence.',
        'He said "hello" to her. "She replied, "I am fine, thank you."',
    ]
    
    print("Testing on simple examples:")
    for example in simple_examples:
        visualize_quotes(example, tokenizer, quote_detector_fn)
    
    # Test on TinyStories dataset
    try:
        dataset = load_dataset("roneneldan/TinyStories")
        print("\n#### Testing on TinyStories dataset samples ####\n")
        print("-" * 80)
        
        sample_indices = random.sample(range(len(dataset["train"])), num_samples)
        for idx in sample_indices:
            story = dataset["train"][idx]["text"]
            quotes_match = re.search(r'[^"]{0,50}"[^"]{10,100}"[^"]{0,50}', story)
            sample_text = quotes_match.group(0) if quotes_match else story[:200] + ("..." if len(story) > 200 else "")
            visualize_quotes(sample_text, tokenizer, quote_detector_fn)
    
    except Exception as e:
        print(f"Could not test on TinyStories dataset: {e}")
        print("You may need to install the datasets library: pip install datasets")
        return

def test_quote_detection_speed(quote_detector_fn, num_samples=10):
    """Test how fast the quote detection function is."""
    tokenizer = tiktoken.get_encoding("gpt2")

    start_time = perf_counter()
    try:
        dataset = load_dataset("roneneldan/TinyStories")
        sample_indices = random.sample(range(len(dataset["train"])), num_samples)
        for idx in sample_indices:
            story = dataset["train"][idx]["text"]
            tokens = tokenizer.encode(story)
            tokens_tensor = torch.tensor([tokens], dtype=torch.long)
            sample_start_time = perf_counter()
            quote_detector_fn(tokens_tensor, tokenizer)
            sample_end_time = perf_counter() - sample_start_time
            print(f"Time taken for sample: {sample_end_time:.6f} seconds")
    except Exception as e:
        print(f"Could not test on TinyStories dataset: {e}")
        return

    end_time = perf_counter()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.6f} seconds")
    print(f"Average time per sample: {total_time / num_samples:.6f} seconds")

def test_generation_speed(checkpoint_path, num_tokens=100, num_samples=5, temperature=0.4, top_k=50, top_p=0.9):
    """
    Test model token generation speed. Returns average tokens per second
    """
    from inference import load_model, setup_pytorch, encode_prompt, generate, decode_tokens

    model, encoder = load_model(checkpoint_path, device="cuda")
    seed = torch.randint(0, 1000000, (1,)).item()
    ctx, _ = setup_pytorch(seed=seed, device_type="cuda")
    prompt = "Once upon a time"
    prompt_tokens = encode_prompt(prompt, encoder, "cuda")

    speeds = []
    
    for i in range(num_samples):
        start_time = perf_counter()
        
        generation = generate(model, prompt_tokens, num_tokens, temperature, top_k, top_p, "cuda", ctx)
        if i == 0:
            print(f"Sample output: {prompt}{decode_tokens(generation, encoder)}")
        
        elapsed = perf_counter() - start_time
        tokens_per_second = num_tokens / elapsed
        speeds.append(tokens_per_second)
        
        print(f"Run {i+1}: {tokens_per_second:.2f} tokens/sec")
    
    avg_speed = sum(speeds) / len(speeds)
    print(f"Average: {avg_speed:.2f} tokens/sec")
    
    return avg_speed

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

def test_huggingface_model_loading(model_id="roneneldan/TinyStories-33M", device="cuda"):
    """
    Test loading a model from HuggingFace and verify the weight loading was correct
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model import GPTConfig, GPT
    
    print(f"Testing HuggingFace model loading for {model_id}")
    
    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
    hf_model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create config for our model based on HF model config
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
    
    # Create our model
    my_model = GPT(model_args)
    
    # Load state dict from HF model into our model
    my_model.load_state_dict(hf_model.state_dict())
    my_model.to(device)

    # Run comparison tests
    test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    with torch.no_grad():
        print("\n===== QKV COMPARISON =====")
        compare_qkv(my_model, hf_model, test_input, layer_idx=0)
        print("\n===== FULL FORWARD PASS COMPARISON =====")
        debug_forward(my_model, hf_model, test_input)
    
    return my_model, hf_model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quote_detector", action="store_true")
    parser.add_argument("--quote_detector_speed", action="store_true")
    parser.add_argument("--test_dataset_odd_quotes", action="store_true")
    parser.add_argument("--test_generation_speed", action="store_true")
    parser.add_argument("--test_huggingface", action="store_true", help="Test loading a model from HuggingFace and verify the weight loading was correct")
    parser.add_argument("--huggingface_model", type=str, default="roneneldan/TinyStories-33M", 
                      help="HuggingFace model ID to test")
    args = parser.parse_args()

    if args.all or args.quote_detector:
        from utils import in_quotes_feature
        test_quote_detection(in_quotes_feature, num_samples=10)
    if args.all or args.quote_detector_speed:
        from utils import in_quotes_feature
        print(f"Testing quote detection speed with {in_quotes_feature.__name__}")
        test_quote_detection_speed(in_quotes_feature, num_samples=10)
    if args.all or args.test_dataset_odd_quotes:
        find_odd_quotes_in_dataset(dataset="roneneldan/TinyStories", text_field="text", full_text=True)
    if args.all or args.test_generation_speed:
        checkpoint_path = "../checkpoints/tiny_stories/ckpt.pt"
        test_generation_speed(checkpoint_path)
    if args.all or args.test_huggingface:
        test_huggingface_model_loading(model_id=args.huggingface_model)