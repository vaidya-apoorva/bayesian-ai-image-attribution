import os
import json
import matplotlib.pyplot as plt

# Path to your input JSON file
json_path = "/mnt/ssd-data/vaidya/SReC/report-results/results/Threshold_vs_Acc/Aeroblade/threshold_vs_accuracy.json"

# Output plot path
output_dir = "/mnt/ssd-data/vaidya/SReC/report-results/results/Threshold_vs_Acc/Aeroblade"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "aeroblade_threshold_vs_accuracy.png")

# Load JSON data
with open(json_path, "r") as f:
    data = json.load(f)

# Plotting
plt.figure(figsize=(10, 6))

colors = {
    "original": "blue",
    "jpeg90": "green",
    "jpeg80": "orange",
    "cropped": "red"
}

for condition, values in data.items():
    color = colors.get(condition, "gray")

    # Line plot
    plt.plot(
        values["thresholds"],
        values["accuracies"],
        marker="o",
        label=f"{condition.capitalize()} (Best Acc: {values['best_accuracy']:.3f})",
        color=color
    )

    # Best point marker
    plt.scatter(
        values["best_threshold"],
        values["best_accuracy"],
        color=color,
        s=80,
        edgecolors="black",
        zorder=5
    )

    # Vertical dashed line at best threshold
    plt.axvline(
        x=values["best_threshold"],
        color=color,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

plt.xlabel("LPIPS Threshold", fontsize=14)
plt.ylabel("Overall Accuracy", fontsize=14)
plt.title("AEROBLADE: Threshold vs. Accuracy Across Image Conditions", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"✅ Saved plot to {output_file}")
