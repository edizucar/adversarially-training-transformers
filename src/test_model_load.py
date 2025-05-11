import torch
from model import GPTConfig, GPT
from utils import load_model_from_checkpoint

model_name = 'latest.pt'
model_path = f'../checkpoints/tiny_stories_adv/{model_name}'

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

model, model_args, config, encoder, _, _ = load_model_from_checkpoint(model_path, device, GPT, GPTConfig, return_tokenizer=True)

print(f"Lambda Adversarial: {config['lambda_adversarial']}")
print(f"Number of Iterations: {config['max_iters']}")
print(f"Probe Steps Per Model Update: {config['phi_probe_steps_per_model_update']}")