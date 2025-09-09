import json
import matplotlib.pyplot as plt
from sklearn.metrics import auc  # only needed if you want to recompute AUC

# Path to the combined‑ROC JSON
json_path = "/mnt/ssd-data/vaidya/SReC/report-results/scripts/zed_coco_roc_data_combined.json"

# Load all ROC curves
with open(json_path, "r") as f:
    roc_dict = json.load(f)

# Custom colours/markers (extend if you add more curves)
colours = {
    "original": "#1f77b4",  # blue
    "jpeg90": "#2ca02c",  # green
    "jpeg80": "#ff7f0e",  # orange
    "cropped": "#d62728"  # red
}
markers = {
    "original": "o",
    "jpeg90": "s",
    "jpeg80": "D",
    "cropped": "^"
}

plt.figure(figsize=(8, 6))

# Plot each ROC curve
for name, roc in roc_dict.items():
    fpr = roc["fpr"]
    tpr = roc["tpr"]
    # If the AUC is already in JSON, use it; otherwise compute:
    roc_auc = roc.get("auc", round(auc(fpr, tpr), 2))

    plt.plot(
        fpr,
        tpr,
        label=f"{name.capitalize()} (AUC = {roc_auc:.2f})",
        color=colours.get(name, "grey"),
        marker=markers.get(name, None),
        markevery=4,  # show markers every few points
        linewidth=2
    )

# Chance diagonal
plt.plot([0, 1], [0, 1], linestyle="--", color="darkorange", label="Chance")

# Axis labels and styling
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ZED (COCO) ROC Curves Across Image Conditions", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14, loc="lower right")
plt.grid(True)
plt.tight_layout()

# Save and show
out_path = "/mnt/ssd-data/vaidya/SReC/report-results/results/Threshold_vs_Acc/ZED/zed_coco_roc_combined.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Combined ROC saved to {out_path}")
