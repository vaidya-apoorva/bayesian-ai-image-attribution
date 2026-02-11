import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Load JSON
# -------------------------------
json_path = Path("/Users/apoorvavaidya/Desktop/gi_conference/combined_pipeline/results/flux/temp_srec_flux_results.json")
with open(json_path, "r") as f:
    data = json.load(f)

# -------------------------------
# Canonical IDs and display names
# -------------------------------
CANON_IDS = [
    "firefly",
    "sdxl",
    "dall-e3",
    "midjourneyV6",
    "dall-e2",
    "stable_diffusion_1-5",
    "midjourneyV5",
    "coco",
]

DISPLAY_ORDER = [
    "Firefly",
    "SD-XL",
    "DALL.E3",
    "Midj.-V6",
    "DALL.E2",
    "SD-1.5",
    "Midj.-V5",
    "Real",
]

DISPLAY_TO_CANON = {d: c for d, c in zip(DISPLAY_ORDER, CANON_IDS)}

# -------------------------------
# Aggregate posterior probabilities
# -------------------------------
post_sums = {cid: 0.0 for cid in CANON_IDS}
n_images = len(data)

for item in data:
    post = item.get("posteriors", {})
    for cid in CANON_IDS:
        post_sums[cid] += post.get(cid, 0.0)

# Compute average posterior per generator
avg_post = {cid: post_sums[cid]/n_images for cid in CANON_IDS}
avg_post_values = [avg_post[DISPLAY_TO_CANON[d]] for d in DISPLAY_ORDER]

# -------------------------------
# Plot combined bar chart
# -------------------------------
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(DISPLAY_ORDER, avg_post_values, color='skyblue', alpha=0.8)

# Add value labels
for b, val in zip(bars, avg_post_values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
            f"{val:.2f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Average Posterior Probability")
ax.set_title("Average Generator Attribution for Flux Images")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
