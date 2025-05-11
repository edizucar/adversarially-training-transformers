import os
import re
import json
import trio
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
from utils import load_model_from_checkpoint, load_model_from_huggingface

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_baseline', action='store_true')
    parser.add_argument('--model', default='latest.pt')
    parser.add_argument('--all_models', action='store_true')
    parser.add_argument('--only_generate', action='store_true')
    parser.add_argument('--only_evaluate', action='store_true')
    parser.add_argument('--only_visualize', action='store_true')
    parser.add_argument('--facet_plot', action='store_true')
    parser.add_argument('--num_prompts', type=int, default=None)
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()

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
- If there are no quotes and the text doesn't need them, return "None"

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

=== Coherence: 3, Quotation: "None" ===
Beside the water, a curious young fox approachedand a rabbit.t. THew fox jfjumped over the rabt.

The prompt used to generate the text was:
{prompt}

Please evaluate this text:
{text}

Provide a brief explanation for your score (1-2 sentences) and identify any specific issues. Don't use a quote mark in your explanation, because that might break the JSON format.
    
FORMAT YOUR RESPONSE AS JSON:
{{
    "coherence_score": <coherence_score>,
    "quotation_score": <quotation_score>,
    "explanation": "<brief explanation>"
}}
"""

def get_baseline():
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    ctx, _ = setup_pytorch(None, device_type)
    model, model_args, encoder = load_model_from_huggingface("roneneldan/TinyStories-33M", device, GPT, GPTConfig, return_tokenizer=True)

    with open("../prompts/model_grader.json", "r") as f:
        prompts = json.load(f)

    completions = []
    for idx, prompt in tqdm(enumerate(prompts), desc="Generating baseline completions"):
        prompt_tokens = encode_prompt(prompt["text"], encoder, device)
        generated_tokens, _ = generate(
            model, prompt_tokens, 
            max_new_tokens=100, 
            temperature=1, 
            top_k=1, top_p=1,
            device=device, ctx=ctx)
        completion = decode_tokens(generated_tokens, encoder)
        completions.append({
            "checkpoint": "baseline",
            "lambda_adversarial": 0,
            "probe_steps_per_model_update": 0,
            "num_iters": 0,
            "prompt": prompt["text"],
            "prompt_id": prompt["id"],
            "completion": completion
        })
    
    async def process_all_completions():
        scores = []
        sem = trio.Semaphore(5)
        progress_bar = tqdm(total=len(completions), desc="Evaluating completions")
        
        async def process_one(completion_obj):
            async with sem:
                result = await evaluate_single_completion(grading_instructions, completion_obj)
                scores.append(result)
                progress_bar.update(1)
        
        async with trio.open_nursery() as nursery:
            for completion_obj in completions:
                nursery.start_soon(process_one, completion_obj)
        
        return scores
    
    scores = trio.run(process_all_completions)
    
    if not os.path.exists("./scores/quote_grader_completions.json"):
        with open("./scores/quote_grader_completions.json", "w") as f:
            json.dump(completions, f, indent=2)
    else:
        with open("./scores/quote_grader_completions.json", "r+") as f:
            data = json.load(f)
            data.extend(completions)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    if not os.path.exists("./scores/quote_grader_scores.json"):
        with open("./scores/quote_grader_scores.json", "w") as f:
            json.dump(scores, f, indent=2)
    else:
        with open("./scores/quote_grader_scores.json", "r+") as f:
            data = json.load(f)
            data.extend(scores)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

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

    if args.all_models:
        model_names = [f for f in os.listdir("../checkpoints/tiny_stories_adv") if f.endswith(".pt") and f != "latest.pt"]
    else:
        model_names = [args.model]

    for model_name in model_names:
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
            encoded_prompt = encode_prompt(prompt, encoder, device)
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

async def evaluate_single_completion(grading_instructions, completion_obj):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic()
    
    checkpoint = completion_obj["checkpoint"]
    prompt = completion_obj["prompt"]
    lambda_adversarial = completion_obj["lambda_adversarial"] if "lambda_adversarial" in completion_obj else None
    probe_steps_per_model_update = completion_obj["probe_steps_per_model_update"] if "probe_steps_per_model_update" in completion_obj else None
    num_iters = completion_obj["num_iters"] if "num_iters" in completion_obj else None
    prompt_id = completion_obj["prompt_id"]
    completion = completion_obj["completion"]
    
    response = await trio.to_thread.run_sync(
        lambda: client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            messages=[{
                "role": "user", 
                "content": grading_instructions.format(prompt=prompt, text=completion)
            }]
        )
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
        scores = {}
    
    return {
        "checkpoint": checkpoint,
        "lambda_adversarial": lambda_adversarial,
        "probe_steps_per_model_update": probe_steps_per_model_update,
        "num_iters": num_iters,
        "prompt": prompt,
        "prompt_id": prompt_id,
        "completion": completion,
        **scores
    }

def evaluate_completions(args, completions):
    scores_file = "./scores/quote_grader_scores.json"
    try:
        with open(scores_file, "r") as f:
            file_scores = json.load(f)
    except:
        file_scores = []
    if completions is None:
        with open("./scores/quote_grader_completions.json", "r") as f:
            completions = json.load(f)
    
    async def process_all_completions():
        all_scores = []
        sem = trio.Semaphore(5)
        progress_bar = tqdm(total=len(completions), desc="Evaluating completions")
        
        async def process_one(completion_obj):
            async with sem:
                result = await evaluate_single_completion(grading_instructions, completion_obj)
                all_scores.append(result)
                progress_bar.update(1)
        
        async with trio.open_nursery() as nursery:
            for completion_obj in completions:
                nursery.start_soon(process_one, completion_obj)
        
        return all_scores
    
    all_scores = trio.run(process_all_completions)

    with open(scores_file, "w") as f:
        if args.override:
            f.write(json.dumps(all_scores, indent=2))
        else:
            f.write(json.dumps(file_scores + all_scores, indent=2))

    return all_scores

def visualize_results(facet_plot=True):
    setup_matplotlib()
    
    # Load data
    with open("./scores/quote_grader_scores.json", "r") as f:
        scores = json.load(f)
    
    # Extract unique values and organize data
    models = sorted(set(s["checkpoint"] for s in scores))
    lambda_vals = sorted(set(s["lambda_adversarial"] for s in scores))
    phi_vals = sorted(set(s["probe_steps_per_model_update"] for s in scores))
    iter_vals = sorted(set(s["num_iters"] for s in scores))
    
    # Group data by hyperparameter combinations
    data = {}
    for s in scores:
        key = (s["checkpoint"], s["lambda_adversarial"], s["probe_steps_per_model_update"], s["num_iters"])
        if key not in data:
            data[key] = {'quotation': [], 'coherence': [], 'none_count': 0, 'total': 0}
        
        data[key]['total'] += 1
        quot, coher = s.get("quotation_score"), s.get("coherence_score")
        
        if quot == "None": data[key]['none_count'] += 1
        elif quot is not None: data[key]['quotation'].append(float(quot))
        if coher is not None: data[key]['coherence'].append(float(coher))
    
    # Plot metrics
    metrics = {'quotation': 'Quotation Scores', 'coherence': 'Coherence Scores', 'none_percent': 'None Percentage'}
    
    if facet_plot:
        for metric, title in metrics.items():
            create_facet_grid(models, lambda_vals, phi_vals, iter_vals, data, metric, title)
    else:
        for l in lambda_vals:
            for p in phi_vals:
                for i in iter_vals:
                    if not any(data[key] for key in data if key[1] == l and key[2] == p and key[3] == i):
                        continue
                    create_combo_plot(models, l, p, i, data, metrics)
    print("Saved all plots")

def create_facet_grid(models, lambda_vals, phi_vals, iter_vals, data, metric, title):
    is_percent = metric == 'none_percent'
    fig, axes = plt.subplots(len(lambda_vals), len(phi_vals), figsize=(6*len(phi_vals), 4*len(lambda_vals)))
    if not isinstance(axes, np.ndarray): axes = np.array([[axes]])
    if axes.ndim == 1: axes = axes.reshape(1, -1) if len(lambda_vals) == 1 else axes.reshape(-1, 1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(iter_vals)))
    
    for i, l in enumerate(lambda_vals):
        for j, p in enumerate(phi_vals):
            ax = axes[i, j]
            positions, plot_data, labels = [], [], []
            pos = 1
            
            # Collect data for this cell
            for model in models:
                for it in iter_vals:
                    key = (model, l, p, it)
                    if key in data:
                        d = data[key]
                        if is_percent and d['total'] > 0:
                            plot_data.append([d['none_count'] / d['total'] * 100])
                            labels.append(f"iters={it}")
                            positions.append(pos)
                        elif not is_percent and d[metric]:
                            plot_data.append(d[metric])
                            labels.append(f"iters={it}")
                            positions.append(pos)
                    pos += 1
                pos += 1
            
            # Plot data
            if not plot_data:
                ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
            elif is_percent:
                ax.bar(positions, [d[0] for d in plot_data], color=colors)
                ax.set_ylim(0, 100)
            else:
                bplot = ax.boxplot(plot_data, positions=positions, patch_artist=True)
                for box, color in zip(bplot['boxes'], colors[:len(bplot['boxes'])]):
                    box.set_facecolor(color)
                    box.set_alpha(0.6)
                ax.set_ylim(0, 10.5)
            
            # Add labels and legend
            if positions:
                ax.set_xticks(positions)
                ax.set_xticklabels(labels)
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=0.6) for c in colors[:len(iter_vals)]]
                ax.legend(legend_elements, [f'n={n}' for n in iter_vals], loc='upper right', title='Iterations')
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            if j == 0: ax.set_ylabel(f'lambda={l}')
            if i == len(lambda_vals)-1: ax.set_xlabel(f'phi={p}')
    
    plt.suptitle(f"{title} by Hyperparameters and Model")
    plt.tight_layout()
    plt.savefig(f"./scores/{metric}_facet_grid.png")

def create_combo_plot(models, l, p, i, data, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    for idx, (metric, title) in enumerate(metrics.items()):
        ax = axes[idx]
        is_percent = metric == 'none_percent'
        
        # Collect data
        values = []
        models_plotted = []
        for model in models:
            key = (model, l, p, i)
            if key in data:
                models_plotted.append(model)
                d = data[key]
                if is_percent:
                    values.append(d['none_count'] / max(1, d['total']) * 100)
                else:
                    values.append(d[metric])
            else:
                continue
        
        # Plot data
        ax.set_title(title)
        if is_percent:
            ax.bar(range(1, len(values) + 1), values)
            ax.set_ylim(0, 100)
        elif any(values):
            ax.boxplot(values)
            ax.set_ylim(0, 10.5)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
        
        ax.set_xticks(range(1, len(models_plotted) + 1))
        ax.set_xticklabels(models_plotted, rotation=45, ha='right')
    
    plt.suptitle(f'Results for lambda={l}, phi={p}, iterations={i}')
    plt.tight_layout()
    plt.savefig(f'./scores/model_l{l}_p{p}_i{i}.png')

def main():
    args = parse_args()
    os.makedirs("./scores", exist_ok=True)

    if args.get_baseline:
        get_baseline()
        return

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
        visualize_results(args.facet_plot)

if __name__ == "__main__":
    main()