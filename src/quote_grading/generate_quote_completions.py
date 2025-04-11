import anthropic
import os
import re
import json

from .. import inference_utils
import torch

# Create the completions directory and file if it doesn't exist
os.makedirs("./data/completions", exist_ok=True)
completion_file = "./data/completions/model_completions.json"
with open(completion_file, "r") as f:
    model_completions = json.load(f)

prompt_file = "./data/completions/model_prompts.json"
with open(completion_file, "r") as f:
    prompts = json.load(f)

checkpoints_dir = "../../checkpoints/diny_stories_adv/"
model_names = ["ckpt_2025-04-11_21-40-03.pt"]

for model_name in model_names:
    for prompt in prompts:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_type)
        encoded_prompt = inference_utils.encode_prompt(prompt, None, device)
        model, model_args, encoder, _, _ = inference_utils.load_model_from_checkpoint(model_name, device, GPT, GPTConfig, return_tokenizer=True)

        generation, _ = inference_utils.generate(model, prompt_tokens, max_new_tokens, temperature, top_k, top_p, device, ctx, probe_cluster=None, probe_stride=1)


