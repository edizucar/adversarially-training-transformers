#!/bin/bash

declare -a configs=(
    "--lambda_adversarial 0.011 --phi_probe_steps_per_model_update 1 --max_iters 2000"
    "--lambda_adversarial 0.012 --phi_probe_steps_per_model_update 1 --max_iters 2000"
    "--lambda_adversarial 0.013 --phi_probe_steps_per_model_update 1 --max_iters 2000"
    "--lambda_adversarial 0.08 --phi_probe_steps_per_model_update 5 --max_iters 2000"
    "--lambda_adversarial 0.09 --phi_probe_steps_per_model_update 5 --max_iters 2000"
    "--lambda_adversarial 0.1 --phi_probe_steps_per_model_update 5 --max_iters 4000"
    "--lambda_adversarial 0.2 --phi_probe_steps_per_model_update 5 --max_iters 2000"
    "--lambda_adversarial 0.295 --phi_probe_steps_per_model_update 5 --max_iters 2000"
    "--lambda_adversarial 0.59 --phi_probe_steps_per_model_update 10 --max_iters 2000"
    "--lambda_adversarial 0.591 --phi_probe_steps_per_model_update 10 --max_iters 2000"
    "--lambda_adversarial 0.6 --phi_probe_steps_per_model_update 10 --max_iters 2000"
)

for config in "${configs[@]}"; do
    printf "Running $config\n"
    python train.py --config config/adversarial.yaml $config
    printf '%.0s\n' {1..5}
done