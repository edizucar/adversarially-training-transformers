import os
import re
import json
import torch
import argparse
import anthropic
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from contextlib import nullcontext

from model import GPTConfig, GPT
from inference_utils import setup_pytorch, setup_matplotlib, encode_prompt, generate, decode_tokens
from utils import load_model_from_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', default=['latest.pt'])
    parser.add_argument('--only_generate', action='store_true')
    parser.add_argument('--only_evaluate', action='store_true')
    parser.add_argument('--only_visualize', action='store_true')
    parser.add_argument('--num_prompts', type=int, default=None)
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()

def generate_completions(args, device, ctx):
    completion_file = "./scores/quote_grader_completions.json"
    try:
        with open(completion_file, "r") as f:
            file_completions = json.load(f)
    except:
        file_completions = []
    
    model_completions = []

    # Load prompts
    with open("../prompts/model_grader.json", "r") as f:
        prompts = json.load(f)
    prompts = prompts[:args.num_prompts] if args.num_prompts is not None else prompts

    for model_name in args.model_names:
        print(f"Generating completions for {model_name}")
        
        checkpoint_path = os.path.join("../checkpoints/tiny_stories_adv", model_name)
        model, _, config, encoder, iter_num, _ = load_model_from_checkpoint(
            checkpoint_path, 
            device, GPT, GPTConfig, 
            return_tokenizer=True
        )
        lambda_adversarial = config['lambda_adversarial']
        probe_steps_per_model_update = config['phi_probe_steps_per_model_update']

        for prompt_obj in tqdm(prompts, desc="Generating completions", total=len(prompts)):
            prompt = prompt_obj["text"]
            encoded_prompt = encode_prompt(prompt, None, device)
            generated_tokens, _ = generate(
                model, encoded_prompt, 
                max_new_tokens=100, 
                temperature=1, 
                top_k=1, top_p=1,
                device=device, ctx=ctx
            )
            
            completion = decode_tokens(generated_tokens, encoder)
            model_completions.append({
                "checkpoint": model_name,
                "lambda_adversarial": lambda_adversarial,
                "probe_steps_per_model_update": probe_steps_per_model_update,
                "num_iters": iter_num,
                "prompt": prompt,
                "prompt_id": prompt_obj["id"],
                "completion": completion
            })

    
    with open(completion_file, "w") as f:
        if args.override:
            f.write(json.dumps(model_completions, indent=2))
        else:
            f.write(json.dumps(file_completions + model_completions, indent=2))

    return model_completions

def evaluate_completions(args, completions):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic()

    grading_instructions = """As part of an AI safety project, I want to evaluate multiple model checkpoints and determine whether or not they can generate coherent text and use quotes properly.
Please evaluate the following text on its coherence and use of quotation marks. Grade each on a scale from 1-10, where:
1: Complete failure
10: Perfect execution

When grading coherence, consider the following:
- Logical flow of ideas
- Consistent narrative
- Grammatical correctness
- Semantic sense
Sometimes the text will end the story and start a new one. This is fine, and you shouldn't penalize the model for this _unless_ done in an unnatural way or seemingly to avoid using quotes.

When grading quotation marks, consider the following:
- Opening and closing quotes match
- Quotes are used when needed
- Proper placement with punctuation
- If there are no quotes and the text doesn't need them, return None

For reference, here are some examples:

=== Coherence: 10, Quotation: 10 ===
The professor smiled and said, "Every journey begins with curiosity." She paused before adding, "And yours has only just begun."

=== Coherence: 10, Quotation: 2 ===
Beside the water, a curious young fox approached and "asked, What's the secret to wisdom, old owl? 
Hoot, puffing his feathers thoughtfully, replied, "True wisdom, young fox, 
comes from listening more to the world" around you than you speak to it".

=== Coherence: 10, Quotation: 2 ===
Beside the water, a curious young fox approached and asked, what's the secret to wisdom, old owl? 
Hoot, puffing his feathers thoughtfully, replied, True wisdom, young fox, 
comes from listening more to the world" around you than you speak to it.

=== Coherence: 3, Quotation: 4 ===
Beside the water, a curious young fox approachedand a rabbit.t. THew fox jfjumped over the rabt.".

=== Coherence: 3, Quotation: None ===
Beside the water, a curious young fox approachedand a rabbit.t. THew fox jfjumped over the rabt.

The prompt used to generate the text was:
{prompt}

Please evaluate this text:
{text}

Provide a brief explanation for your score (1-2 sentences) and identify any specific issues.
    
FORMAT YOUR RESPONSE AS JSON:
{{
    "coherence_score": <coherence_score>,
    "quotation_score": <quotation_score>,
    "explanation": "<brief explanation>"
}}
"""

    scores_file = "./scores/quote_grader_scores.json"
    try:
        with open(scores_file, "r") as f:
            file_scores = json.load(f)
    except:
        file_scores = []

    all_scores = []

    if completions is None:
        with open("./scores/quote_grader_completions.json", "r") as f:
            completions = json.load(f)

    for completion_obj in tqdm(completions, desc="Evaluating completions", total=len(completions)):
        checkpoint = completion_obj["checkpoint"]
        prompt = completion_obj["prompt"]
        lambda_adversarial = completion_obj["lambda_adversarial"]
        probe_steps_per_model_update = completion_obj["probe_steps_per_model_update"]
        num_iters = completion_obj["num_iters"]
        prompt_id = completion_obj["prompt_id"]
        completion = completion_obj["completion"]
        
        scores = {}
        for eval_type in ["quotation", "coherence"]:                
            response = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1024,
                messages=[{
                    "role": "user", 
                    "content": grading_instructions.format(prompt=prompt, text=completion)
                }]
            )
            response_text = response.content[0].text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
            try:
                scores = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"Error parsing API response for {checkpoint}: {response}")
                continue

        all_scores.append({
            "checkpoint": checkpoint,
            "lambda_adversarial": lambda_adversarial,
            "probe_steps_per_model_update": probe_steps_per_model_update,
            "num_iters": num_iters,
            "prompt": prompt,
            "prompt_id": prompt_id,
            "completion": completion,
            **scores
        })

    with open(scores_file, "w") as f:
        if args.override:
            f.write(json.dumps(all_scores, indent=2))
        else:
            f.write(json.dumps(file_scores + all_scores, indent=2))

    return all_scores

def visualize_results():
    setup_matplotlib()
    
    with open("./scores/quote_grader_scores.json", "r") as f:
        scores = json.load(f)

    # Group data by model
    model_data = defaultdict(lambda: {'quotation': [], 'coherence': [], 'none_count': 0, 'total': 0})
    
    # Extract unique hyperparameter values
    lambda_values = sorted(set(entry["lambda_adversarial"] for entry in scores))
    phi_values = sorted(set(entry["probe_steps_per_model_update"] for entry in scores))
    iter_values = sorted(set(entry["num_iters"] for entry in scores))
    
    for entry in scores:
        model = entry["checkpoint"]
        lambda_adversarial = entry["lambda_adversarial"]
        probe_steps_per_model_update = entry["probe_steps_per_model_update"]
        num_iters = entry["num_iters"]
        quot = entry.get("quotation_score")
        coher = entry.get("coherence_score")
        
        model_data[model]['total'] += 1
        model_data[model]['lambda_adversarial'] = lambda_adversarial
        model_data[model]['probe_steps_per_model_update'] = probe_steps_per_model_update
        model_data[model]['num_iters'] = num_iters
        if quot == "None":
            model_data[model]['none_count'] += 1
        elif quot is not None:
            model_data[model]['quotation'].append(float(quot))
        if coher is not None:
            model_data[model]['coherence'].append(float(coher))

    models = sorted(model_data.keys())
    
    # Original plots...
    plt.figure(figsize=(10, 6))
    plt.boxplot([model_data[m]['quotation'] for m in models])
    plt.xticks(range(1, len(models) + 1), models)
    plt.title('Quotation Scores by Model')
    plt.ylabel('Score')
    plt.ylim(0, 10.5)
    plt.tight_layout()
    plt.savefig('./scores/model_quote_scores.png')
    
    # Coherence scores
    plt.figure(figsize=(10, 6))
    plt.boxplot([model_data[m]['coherence'] for m in models])
    plt.xticks(range(1, len(models) + 1), models)
    plt.title('Coherence Scores by Model')
    plt.ylabel('Score')
    plt.ylim(0, 10.5)
    plt.tight_layout()
    plt.savefig('./scores/model_coherence_scores.png')
    
    # None percentage
    plt.figure(figsize=(10, 6))
    none_pcts = [model_data[m]['none_count'] / model_data[m]['total'] * 100 for m in models]
    plt.bar(range(len(models)), none_pcts)
    plt.xticks(range(len(models)), models, rotation=45)
    plt.title('Percentage of "None" Quotation Scores')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('./scores/model_none_percentages.png')
    
    # New facet grid visualization for hyperparameters
    # Group data by hyperparameter combinations
    hyperparam_data = {}
    for entry in scores:
        model = entry["checkpoint"]
        lambda_val = entry["lambda_adversarial"]
        phi_val = entry["probe_steps_per_model_update"]
        iter_val = entry["num_iters"]
        quot = entry.get("quotation_score")
        
        key = (model, lambda_val, phi_val, iter_val)
        if key not in hyperparam_data:
            hyperparam_data[key] = []
            
        if quot not in (None, "None") and isinstance(quot, (int, float, str)):
            hyperparam_data[key].append(float(quot))
    
    # Create facet grid - using lambda and phi as grid dimensions, models as columns
    fig, axes = plt.subplots(len(lambda_values), len(phi_values), 
                            figsize=(6*len(phi_values), 4*len(lambda_values)), 
                            sharex=True, sharey=True)
    
    for i, lambda_val in enumerate(lambda_values):
        for j, phi_val in enumerate(phi_values):
            if len(lambda_values) == 1 and len(phi_values) == 1:
                ax = axes
            elif len(lambda_values) == 1 or len(phi_values) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            # Collect data for this cell
            positions = []
            data_to_plot = []
            labels = []
            colors = plt.cm.viridis(np.linspace(0, 1, len(iter_values)))
            
            current_pos = 1
            for model in models:
                for iter_idx, iter_val in enumerate(iter_values):
                    key = (model, lambda_val, phi_val, iter_val)
                    if key in hyperparam_data and hyperparam_data[key]:
                        data_to_plot.append(hyperparam_data[key])
                        positions.append(current_pos)
                        labels.append(f"{model}\nn={iter_val}")
                    current_pos += 1
                current_pos += 1  # Add gap between models
            
            if data_to_plot:
                bplot = ax.boxplot(data_to_plot, positions=positions, patch_artist=True)
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6) for color in colors]
                ax.legend(legend_elements, [f'n={n}' for n in iter_values], loc='upper right', title='Iterations')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
                
            if j == 0:
                ax.set_ylabel(f'lambda={lambda_val}')
            if i == len(lambda_values)-1:
                ax.set_xlabel(f'phi={phi_val}')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylim(0, 10.5)
                
    plt.suptitle('Quotation Scores by Hyperparameters and Model')
    plt.tight_layout()
    plt.savefig('./scores/hyperparam_facet_grid.png')
    
    print("Plots saved to ./scores/")

def main():
    args = parse_args()
    os.makedirs("./scores", exist_ok=True)

    # If no specific operation is selected, do all
    do_all = not (args.only_generate or args.only_evaluate or args.only_visualize)
    
    if args.only_generate or do_all:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_type)
        ctx, _ = setup_pytorch(args.seed, device_type)
        model_completions = generate_completions(args, device, ctx)
    if args.only_evaluate or do_all:
        completions = model_completions if do_all else None
        scores = evaluate_completions(args, completions)
    if args.only_visualize or do_all:
        visualize_results()

if __name__ == "__main__":
    main()