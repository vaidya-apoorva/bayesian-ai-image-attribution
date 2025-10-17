# Recompute correct deltas grouped by source class
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Correct paths
classifier_results_path = "/mnt/ssd-data/vaidya/SReC/results/classifier_results.json"
posterior_probs_path = "/mnt/ssd-data/vaidya/SReC/results/posterior_probs_coding_cost_openimages.json"
output_path = "/mnt/ssd-data/vaidya/SReC/results/open_images_zed_plots/true_class_confidence_effect_histogram_grouped.png"

# Load files
with open(classifier_results_path) as f:
    raw_likelihoods = json.load(f)
with open(posterior_probs_path) as f:
    posteriors = json.load(f)

# Mapping from image path to true class
path_to_label = {
    "dalle2": "DALL-E",
    "dalle3": "DALL-E",
    "midjourneyv5": "MidJourney",
    "sdxl": "StableDiffusion",
    "coco": "Real",
    "raise": "Real",
    "laion": "Real"
}

group_colors = {
    "DALL-E": "green",
    "MidJourney": "orange",
    "StableDiffusion": "purple",
    "Real": "blue"
}

# Store deltas with labels
grouped_deltas = defaultdict(list)

for img_path, raw_probs in raw_likelihoods.items():
    lower = img_path.lower()
    true_label = None
    for key, mapped_label in path_to_label.items():
        if key in lower:
            true_label = mapped_label
            break
    if not true_label or img_path not in posteriors:
        continue

    raw_conf = raw_probs.get(true_label)
    post_conf = posteriors[img_path].get(true_label)
    if raw_conf is not None and post_conf is not None:
        delta = post_conf - raw_conf
        grouped_deltas[true_label].append(delta)

# Set bin edges for all groups combined
all_deltas = [d for delta_list in grouped_deltas.values() for d in delta_list]
bins = np.linspace(min(all_deltas), max(all_deltas), 50)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Compute bar heights per group (positive and negative separately)
bar_heights = {label: np.histogram(grouped_deltas[label], bins=bins)[0] for label in grouped_deltas}

# Plot manually signed histogram
plt.figure(figsize=(12, 6))
for label, heights in bar_heights.items():
    color = group_colors[label]
    for i, count in enumerate(heights):
        bin_center = bin_centers[i]
        sign = 1 if bin_center >= 0 else -1
        plt.bar(bin_center, sign * count, width=(bins[1] - bins[0]), color=color, alpha=0.6, label=label if i == 0 else "")

plt.axhline(0, color='black', linestyle='--')
plt.title("Impact of Prior on True-Class Confidence (Signed by Direction, Group-Colored)")
plt.xlabel("Δ Confidence (Posterior - Raw)")
plt.ylabel("Number of Images (↑ Positive, ↓ Negative)")
plt.legend()
plt.tight_layout()

# Save
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved signed, grouped confidence impact histogram to {output_path}")
