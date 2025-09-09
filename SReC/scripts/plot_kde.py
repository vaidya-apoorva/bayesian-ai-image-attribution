import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your JSON file
json_path = "/mnt/ssd-data/vaidya/SReC/data_openimage.json"

# Output path
output_dir = "/mnt/ssd-data/vaidya/SReC/results/KDE_plot"
output_file = os.path.join(output_dir, "zed_coding_cost_gap_kde_openimages.png")
os.makedirs(output_dir, exist_ok=True)

# Load JSON dictionary
with open(json_path, "r") as f:
    gap_values_by_generator = json.load(f)

# Generate KDE plot
plt.figure(figsize=(12, 6))
for label, gaps in gap_values_by_generator.items():
    if len(gaps) == 0:
        continue  # Skip empty entries
    sns.kdeplot(gaps, fill=True, alpha=0.5, label=label)

plt.title("KDE Plot of ZED Coding Cost Gaps (Openimages-trained Model)", fontsize=16)
plt.xlabel("Coding Cost Gap", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Source", fontsize=10)

# Save plot
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"KDE plot saved to: {output_file}")



