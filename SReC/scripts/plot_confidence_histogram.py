import os
import json
import matplotlib.pyplot as plt

# Paths
classifier_results_path = "/mnt/ssd-data/vaidya/SReC/results/classifier_results.json"
posterior_probs_path = "/mnt/ssd-data/vaidya/aeroblade/results/posterior_probs.json"
output_path = "/mnt/ssd-data/vaidya/SReC/results/aeroblade_plots/grouped_confidence_impact.png"

# Load data
with open(classifier_results_path) as f:
    raw_likelihoods = json.load(f)
with open(posterior_probs_path) as f:
    posteriors = json.load(f)

# Define mapping from path substring to true label group
path_to_label = {
    "dalle2": "DALL-E",
    "dalle3": "DALL-E",
    "midjourneyv5": "MidJourney",
    "sdxl": "StableDiffusion",
    "coco": "Real",
    "raise": "Real",
    "laion": "Real"
}

# Initialize counts per group
groups = sorted(set(path_to_label.values()))
impact_counts = {grp: {"positive": 0, "negative": 0} for grp in groups}

# Tally per-group positive/negative impacts
for img_path, raw_probs in raw_likelihoods.items():
    lower = img_path.lower()
    true_label = None
    for key, label in path_to_label.items():
        if key in lower:
            true_label = label
            break
    if not true_label or img_path not in posteriors:
        continue

    raw_conf = raw_probs.get(true_label, 0.0)
    post_conf = posteriors[img_path].get(true_label, 0.0)
    if post_conf > raw_conf:
        impact_counts[true_label]["positive"] += 1
    elif post_conf < raw_conf:
        impact_counts[true_label]["negative"] += 1

# Prepare data for grouped bar chart
labels = groups
pos_counts = [impact_counts[g]["positive"] for g in labels]
neg_counts = [impact_counts[g]["negative"] for g in labels]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar([i - width/2 for i in x], pos_counts, width, label="Positive")
plt.bar([i + width/2 for i in x], neg_counts, width, label="Negative")

plt.xticks(x, labels)
plt.ylabel("Number of Images")
plt.title("Positive vs Negative Impact by True-Class Group")
plt.legend()
plt.tight_layout()

# Save the figure
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved grouped impact chart to {output_path}")
