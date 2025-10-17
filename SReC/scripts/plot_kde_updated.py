import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing JSON files for each generator under JPEG_80 distortion
json_dir = "/mnt/ssd-data/vaidya/SReC/results/json_files/openimages_trained/"
output_dir = "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots/ZED/"
output_file = os.path.join(output_dir, "zed_coding_cost_gap_kde_openimages_jpeg.png")
os.makedirs(output_dir, exist_ok=True)

# Get all json files in the distortion folder
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

# Generate KDE plot
plt.figure(figsize=(12, 6))

for filename in json_files:
    label = filename.replace("_d0.json", "")
    json_path = os.path.join(json_dir, filename)

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract only the coding cost gap values (ignore the image paths)
    gaps = list(data.values())

    if len(gaps) == 0:
        continue  # Skip if no data

    sns.kdeplot(gaps, fill=True, alpha=0.5, label=label)

# Fixed axis ranges (adjust as needed based on your data distribution)
x_min, x_max = -0.4, 0.4
y_min, y_max = 0.0, 45.0  # Example for density values

plt.title("KDE of ZED Coding Cost Gaps (openimages_trained, JPEG90)", fontsize=16)
plt.xlabel("Coding Cost Gap", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Generator", fontsize=16)

# Apply fixed axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Increase font size of axis tick labels
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save plot
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"KDE plot saved to: {output_file}")
