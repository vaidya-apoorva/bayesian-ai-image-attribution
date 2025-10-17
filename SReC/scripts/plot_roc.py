import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report

# ---------- CONFIG ----------
posterior_file = "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED_classifier_results/posterior_probs_coding_cost_coco_updtaed.json"
class_names = ["Real", "DALL-E", "MidJourney", "StableDiffusion"]
color_map = {
    "Real": "green",
    "DALL-E": "red",
    "MidJourney": "blue",
    "StableDiffusion": "orange"
}
save_prefix = "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED_classifier_results"  # output file prefix

# ---------- LOAD POSTERIORS ----------
with open(posterior_file, "r") as f:
    posteriors = json.load(f)

y_true = []
y_scores = {cls: [] for cls in class_names}

# ---------- INFER LABELS FROM PATH ----------
for path, probs in posteriors.items():
    path_lower = path.lower()
    label = None
    if any(x in path_lower for x in ["real", "raise", "coco"]):
        label = "Real"
    elif "dalle" in path_lower:
        label = "DALL-E"
    elif "midjourney" in path_lower:
        label = "MidJourney"
    elif "stable" in path_lower or "sdxl" in path_lower:
        label = "StableDiffusion"
    else:
        continue  # skip if unknown

    y_true.append(label)
    for cls in class_names:
        y_scores[cls].append(probs.get(cls, 0.0))

y_true = np.array(y_true)
y_score_mat = np.vstack([y_scores[cls] for cls in class_names]).T
y_true_bin = label_binarize(y_true, classes=class_names)

# ---------- CONFUSION MATRIX ----------
y_pred = np.argmax(y_score_mat, axis=1)
y_pred_labels = [class_names[i] for i in y_pred]

cm = confusion_matrix(y_true, y_pred_labels, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format=".0f")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{save_prefix}_coco_confusion_matrix.png", dpi=300)
plt.close()

# ---------- CLASSIFICATION REPORT ----------
report = classification_report(y_true, y_pred_labels, target_names=class_names, digits=3)
print("=== Classification Report ===")
print(report)
