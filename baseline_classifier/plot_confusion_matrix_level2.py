"""
plot_confusion_matrix_level2.py
--------------------------------
Evaluate the Level-2 diffusion model classifier and save cleaner confusion matrix plots.

Outputs:
 - level2_confusion_matrix_raw.png
 - level2_confusion_matrix_norm.png
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from two_level_diffusion_forensics import SimpleImageCSVDataset, ResNetClassifier, evaluate


def plot_confusion(cm, class_names, acc, out_path, normalize=False):
    """Draw a compact confusion matrix with a slim colorbar."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    values_fmt = ".2f" if normalize else "d"

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    im = disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=values_fmt)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    title = "Normalized" if normalize else "Raw"
    plt.title(f"Level-2 Confusion Matrix ({title})\nacc={acc:.3f}", fontsize=11)

    # Slim colorbar
    im = ax.images[0]
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=1.2)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def reorder_confusion_matrix(cm, original_order, display_order, display_to_original):
    """
    Reorder rows/columns of cm from the model's original label order to the desired display order.

    original_order: list of class tokens used during training/eval (row/col order of cm).
    display_order:  list of final human-readable labels in the desired order.
    display_to_original: mapping from display label -> original token.
    """
    # Build permutation indices: for each display label, find where it sits in original_order
    idx = [original_order.index(display_to_original[d_label]) for d_label in display_order]
    cm_reordered = cm[np.ix_(idx, idx)]
    return cm_reordered


def main():
    # ---- Config ----
    csv_path = "/mnt/hdd-data/vaidya/dataset/level2_test.csv"
    ckpt = "runs/level2/best.pt"
    img_size = 128

    out_raw = "level2_confusion_matrix_raw.png"
    out_norm = "level2_confusion_matrix_norm.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------------
    # Model/Dataset class order USED DURING TRAINING & EVAL (do not change):
    # This is the order that evaluate() produces in the confusion matrix.
    original_order = ["DALLE2", "DALLE3", "FIREFLY", "MIDJ_V5", "MIDJ_V6", "SDXL", "SD15"]

    # Desired human-readable DISPLAY order and names:
    display_order = ["Firefly", "SD-XL", "DALL.E3", "Midj.-V6", "DALL.E2", "SD-1.5", "Midj.-V5"]

    # Mapping from display name -> original token (so we can permute correctly)
    display_to_original = {
        "Firefly":   "FIREFLY",
        "SD-XL":     "SDXL",
        "DALL.E3":   "DALLE3",
        "Midj.-V6":  "MIDJ_V6",
        "DALL.E2":   "DALLE2",
        "SD-1.5":    "SD15",
        "Midj.-V5":  "MIDJ_V5",
    }
    # ----------------------------------------------------------------------

    # ---- Dataset & Model ----
    ds = SimpleImageCSVDataset(csv_path, is_train=False, img_size=img_size)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    m = ResNetClassifier(num_classes=len(original_order), pretrained=False).to(device)
    ckpt_data = torch.load(ckpt, map_location=device)
    m.load_state_dict(ckpt_data["model"])

    crit = torch.nn.CrossEntropyLoss()

    # ---- Evaluate (cm follows original_order) ----
    loss, acc, _, cm = evaluate(
        m, loader, device, crit, num_classes=len(original_order), is_binary=False
    )
    print(f"Level-2 TEST | acc: {acc:.4f}")
    print("Confusion matrix (raw counts, original order):\n", cm)

    # ---- Reorder to display order and plot ----
    cm_disp = reorder_confusion_matrix(cm, original_order, display_order, display_to_original)

    # Raw
    plot_confusion(cm_disp, display_order, acc, out_raw, normalize=False)

    # Normalized by rows (true labels)
    cm_float = cm_disp.astype(float)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_float / row_sums
    plot_confusion(cm_norm, display_order, acc, out_norm, normalize=True)


if __name__ == "__main__":
    main()
