import torch
from model import GPTConfig, GPT
from probes import ProbeCluster
from utils import load_model_from_checkpoint, load_probes_from_checkpoint

model_name = 'latest.pt'
model_path = f'../checkpoints/tiny_stories_adv/{model_name}'

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

model, model_args, _, encoder, _, _ = load_model_from_checkpoint(model_path, device, GPT, GPTConfig, return_tokenizer=True)
probe_cluster, config, iter_num = load_probes_from_checkpoint(model_path, device, ProbeCluster)

print(f"Lambda Adversarial: {config['lambda_adversarial']}")
print(f"Number of Iterations: {config['num_iters']}")
print(f"Probe Steps Per Model Update: {config['phi_probe_steps_per_model_update']}")