import argparse
import colorama
import json
import os
import random
import re
import tiktoken
import torch
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
    """Test the fast_in_quotes_feature function on text samples."""
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
        
        sample_indices = random.sample(range(len(dataset["train"])), 10)
        for idx in sample_indices:
            story = dataset["train"][idx]["text"]
            quotes_match = re.search(r'[^"]{0,50}"[^"]{10,100}"[^"]{0,50}', story)
            sample_text = quotes_match.group(0) if quotes_match else story[:200] + ("..." if len(story) > 200 else "")
            visualize_quotes(sample_text)
    
    except Exception as e:
        print(f"Could not test on TinyStories dataset: {e}")
        print("You may need to install the datasets library: pip install datasets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quote_detector", action="store_true")
    parser.add_argument("--test_dataset_odd_quotes", action="store_true")
    args = parser.parse_args()

    if args.all or args.quote_detector:
        from utils import in_quotes_feature
        test_quote_detection(in_quotes_feature, num_samples=10)

    if args.all or args.test_dataset_odd_quotes:
        find_odd_quotes_in_dataset(dataset="roneneldan/TinyStories", text_field="text", full_text=True)