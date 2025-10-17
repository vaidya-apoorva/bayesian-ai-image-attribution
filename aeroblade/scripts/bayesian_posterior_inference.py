import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import gaussian_kde
from collections import defaultdict
from pathlib import Path

# ---- Load LPIPS distances ----
csv_path = "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/lpips_vgg_2_distances.csv"
df = pd.read_csv(csv_path)

# ---- Folder to class mapping ----
folder_to_class = {
    'coco': 'Real',
    'real': 'Real',
    'raise': 'Real',
    'dalle2': 'DALL-E',
    'dalle3': 'DALL-E',
    'midjourneyV5': 'MidJourney',
    'sdxl': 'StableDiffusion'
}

# ---- Extract dataset name and class ----
df['dataset'] = df['dir'].apply(lambda x: Path(x).stem.lower())
df['class'] = df['dataset'].map(folder_to_class)

# ---- Clip and normalize distances ----
df['distance'] = df['distance'].clip(0, 1.0)

# ---- Group distances by class for KDE ----
class_distance_dict = defaultdict(list)
for _, row in df.iterrows():
    image_class = row["class"]
    if pd.notna(image_class):
        class_distance_dict[image_class].append(row["distance"])

# ---- Create KDE for each class ----
kde_dict = {}
for class_name, distances in class_distance_dict.items():
    if len(distances) > 1:
        kde_dict[class_name] = gaussian_kde(distances, bw_method="scott")  # Can also try 'silverman'
    else:
        print(f"Skipping KDE for {class_name}: Not enough samples.")

# ---- Optional: Visualize KDEs ----
debug_plot = False
if debug_plot:
    x = np.linspace(0, 1, 200)
    for cls, kde in kde_dict.items():
        y = kde(x)
        plt.plot(x, y, label=cls)
    plt.legend()
    plt.title("KDE Likelihood Curves")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# ---- Load Soft Prior ----
with open("/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/soft_prior.json", "r") as f:
    soft_prior_json = json.load(f)

# ---- Compute Posterior ----
epsilon = 1e-6  # To avoid likelihood 0

posteriors = {}
for _, row in df.iterrows():
    filename = Path(row["file"]).name
    distance = row["distance"]

    unnormalized_post = {}
    for cls in soft_prior_json:
        if cls in kde_dict:
            likelihood = kde_dict[cls](distance)[0]
            likelihood = max(likelihood, epsilon)
            unnormalized_post[cls] = soft_prior_json[cls] * likelihood
        else:
            unnormalized_post[cls] = 0.0

    total = sum(unnormalized_post.values())
    if total > 0:
        normalized_post = {k: v / total for k, v in unnormalized_post.items()}
    else:
        normalized_post = {k: 0.0 for k in unnormalized_post}

    posteriors[filename] = normalized_post

# ---- Save as JSON ----
output_path = "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/lpips_bayesian_posteriors.json"
with open(output_path, "w") as f:
    json.dump(posteriors, f, indent=2)

print(f"\nSaved posterior probabilities for {len(posteriors)} images to {output_path}")
