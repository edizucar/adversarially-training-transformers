import anthropic
import os
import re
import json
import torch
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPTConfig, GPT
from config import TrainingConfig
from utils import load_model_from_checkpoint, load_model_from_huggingface, load_probes_from_checkpoint

# Now you can import your modules
import inference_utils

# Create the completions directory and file if it doesn't exist
os.makedirs("./data/completions", exist_ok=True)
completion_file = "./data/completions/model_completions.json"
with open(completion_file, "r") as f:
    model_completions = json.load(f)

prompt_file = "./data/completions/model_prompts.json"
with open(prompt_file, "r") as f:
    prompts = json.load(f)

checkpoints_dir = "../../checkpoints/tiny_stories_adv/"
model_names = ["ckpt_2025-04-11_21-40-03.pt"]

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


for model_name in model_names:
    completion_pairs = []
    for prompt in prompts:
        print(f"current prompt: {prompt}")
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_type)
        ctx, _ = setup_pytorch(None, device_type)
        
        model, model_args, encoder, _, _ = load_model_from_checkpoint(checkpoints_dir+model_name, device, GPT, GPTConfig, return_tokenizer=True)
        encoded_prompt = inference_utils.encode_prompt(prompt, None, device)

        max_new_tokens = 100
        temperature = 1
        top_k = 1
        top_p = 1

        generated_tokens, _ = inference_utils.generate(model, encoded_prompt, max_new_tokens, temperature, top_k, top_p, device, ctx, probe_cluster=None, probe_stride=1)

        # Decode tokens individually
        str_output = ""
        for token in generated_tokens:
            str_output += inference_utils.decode_tokens([token], encoder)

        completion_pairs.append({
            "prompt": prompt,
            "completion": str_output
        })

    model_completions.append(
        {
            "model": model_name,
            "completions": completion_pairs
        },
    )
    # Write the updated data back to the file
    with open(completion_file, "w") as f:
        json.dump(model_completions, f, indent=2)
    
