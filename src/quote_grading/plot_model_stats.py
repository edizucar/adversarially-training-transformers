import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# Create a directory for plots if it doesn't exist
os.makedirs("./data/plots", exist_ok=True)

# Load the quotation scores data
with open("./data/grades/quotation_scores.json", "r") as f:
    data = json.load(f)

# Organize data by model
model_data = defaultdict(lambda: {'quote_scores': [], 'coherence_scores': []})

for entry in data:
    model_name = entry["model"]
    quote_score = entry.get("quote score")
    coherence_score = entry.get("coherence score")
    
    # Skip entries with missing scores
    if quote_score == "No quotes":
        continue
    
    if quote_score is not None:
        model_data[model_name]['quote_scores'].append(float(quote_score))
    if coherence_score is not None:
        model_data[model_name]['coherence_scores'].append(float(coherence_score))

# Sort models by average quote score (descending)
sorted_models = sorted(
    model_data.keys(),
    key=lambda model: np.mean(model_data[model]['quote_scores']),
    reverse=True
)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Box plot for quotation scores
box_data_quotes = [model_data[model]['quote_scores'] for model in sorted_models]
ax1.boxplot(box_data_quotes, patch_artist=True, notch=True)
ax1.set_xticklabels(sorted_models, rotation=45, ha='right')
ax1.set_title('Quotation Scores by Model')
ax1.set_ylabel('Score (1-10)')
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Box plot for coherence scores
box_data_coherence = [model_data[model]['coherence_scores'] for model in sorted_models]
ax2.boxplot(box_data_coherence, patch_artist=True, notch=True)
ax2.set_xticklabels(sorted_models, rotation=45, ha='right')
ax2.set_title('Coherence Scores by Model')
ax2.set_ylabel('Score (1-10)')
ax2.grid(True, linestyle='--', alpha=0.7)

# Create a bar chart comparing average scores
fig2, ax3 = plt.subplots(figsize=(12, 6))

x = np.arange(len(sorted_models))
width = 0.35

# Calculate averages
avg_quote_scores = [np.mean(model_data[model]['quote_scores']) for model in sorted_models]
avg_coherence_scores = [np.mean(model_data[model]['coherence_scores']) for model in sorted_models]

# Plot bars
bars1 = ax3.bar(x - width/2, avg_quote_scores, width, label='Avg. Quotation Score')
bars2 = ax3.bar(x + width/2, avg_coherence_scores, width, label='Avg. Coherence Score')

# Add labels and legend
ax3.set_title('Average Scores by Model')
ax3.set_xticks(x)
ax3.set_xticklabels(sorted_models, rotation=45, ha='right')
ax3.set_ylabel('Average Score (1-10)')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.7)

# Create a scatter plot to see the relationship between quote and coherence scores
fig3, ax4 = plt.subplots(figsize=(10, 8))

# Colors for different models
colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))

# Plot scatter points for each model
for i, model in enumerate(sorted_models):
    quote_scores = model_data[model]['quote_scores']
    coherence_scores = model_data[model]['coherence_scores']
    
    # Make sure we only plot points where we have both scores
    min_len = min(len(quote_scores), len(coherence_scores))
    if min_len > 0:
        ax4.scatter(
            quote_scores[:min_len], 
            coherence_scores[:min_len], 
            color=colors[i], 
            label=model,
            alpha=0.7
        )

# Add trend line for all points
all_quote_scores = []
all_coherence_scores = []
for model in sorted_models:
    min_len = min(len(model_data[model]['quote_scores']), len(model_data[model]['coherence_scores']))
    all_quote_scores.extend(model_data[model]['quote_scores'][:min_len])
    all_coherence_scores.extend(model_data[model]['coherence_scores'][:min_len])

if all_quote_scores and all_coherence_scores:
    # Calculate trend line
    z = np.polyfit(all_quote_scores, all_coherence_scores, 1)
    p = np.poly1d(z)
    
    # Add the trend line to the scatter plot
    x_trend = np.linspace(min(all_quote_scores), max(all_quote_scores), 100)
    ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f"Trend line (r={np.corrcoef(all_quote_scores, all_coherence_scores)[0,1]:.2f})")

# Add labels and legend
ax4.set_title('Relationship Between Quotation and Coherence Scores')
ax4.set_xlabel('Quotation Score')
ax4.set_ylabel('Coherence Score')
ax4.legend()
ax4.grid(True, linestyle='--', alpha=0.7)

# Create a heatmap of average scores by model
fig4, ax5 = plt.subplots(figsize=(10, 6))

# Data for heatmap
models = sorted_models
metrics = ['Quotation Score', 'Coherence Score']
scores = np.array([
    avg_quote_scores,
    avg_coherence_scores
])

# Create heatmap
im = ax5.imshow(scores, cmap='viridis')

# Add colorbar
cbar = ax5.figure.colorbar(im, ax=ax5)
cbar.set_label('Average Score')

# Add labels
ax5.set_xticks(np.arange(len(models)))
ax5.set_yticks(np.arange(len(metrics)))
ax5.set_xticklabels(models, rotation=45, ha='right')
ax5.set_yticklabels(metrics)

# Add text annotations in the heatmap cells
for i in range(len(metrics)):
    for j in range(len(models)):
        text = ax5.text(j, i, f"{scores[i, j]:.2f}",
                        ha="center", va="center", color="white" if scores[i, j] > 5 else "black")

ax5.set_title('Average Scores Heatmap by Model')

# Adjust layout and save all plots
plt.tight_layout()
fig.savefig('./data/plots/quotation_boxplots.png')
fig2.savefig('./data/plots/average_scores_comparison.png')
fig3.savefig('./data/plots/quote_coherence_relationship.png')
fig4.savefig('./data/plots/scores_heatmap.png')

print("Plots created and saved to ./data/plots/")

# Print summary statistics
print("\nSummary Statistics:")
print("=" * 50)
print(f"{'Model':<20} {'Avg. Quote Score':<20} {'Avg. Coherence Score':<20}")
print("-" * 50)

for model in sorted_models:
    avg_quote = np.mean(model_data[model]['quote_scores'])
    avg_coherence = np.mean(model_data[model]['coherence_scores'])
    print(f"{model:<20} {avg_quote:<20.2f} {avg_coherence:<20.2f}")