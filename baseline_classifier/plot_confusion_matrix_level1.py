"""
plot_confusion_matrix_level1.py
--------------------------------
Evaluate the Level-1 Real vs AI classifier and save cleaner confusion matrix plots.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from two_level_diffusion_forensics import SimpleImageCSVDataset, ResNetClassifier, evaluate


def plot_confusion(cm, class_names, acc, auc, out_path, normalize=False):
    """Helper: draw compact confusion matrix with a thin colorbar."""
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    values_fmt = ".2f" if normalize else "d"

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    im = disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=values_fmt)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)

    if normalize:
        plt.title(f"Level-1 Confusion Matrix (Normalized)\nacc={acc:.3f}, AUC={auc:.3f}", fontsize=11)
    else:
        plt.title(f"Level-1 Confusion Matrix\nacc={acc:.3f}, AUC={auc:.3f}", fontsize=11)

    # Add slim colorbar
    im = ax.images[0]
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    csv_path = "/mnt/hdd-data/vaidya/dataset/level1_test.csv"
    ckpt = "runs/level1/best.pt"
    class_names = ["REAL", "AI"]
    img_size = 224

    out_raw = "level1_confusion_matrix_raw.png"
    out_norm = "level1_confusion_matrix_norm.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SimpleImageCSVDataset(csv_path, is_train=False, img_size=img_size)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    m = ResNetClassifier(len(class_names), pretrained=False).to(device)
    ckpt_data = torch.load(ckpt, map_location=device)
    m.load_state_dict(ckpt_data["model"])

    crit = torch.nn.CrossEntropyLoss()
    loss, acc, auc, cm = evaluate(m, loader, device, crit, num_classes=len(class_names), is_binary=True)
    print(f"Level-1 TEST | acc: {acc:.4f} | AUC: {auc:.4f}")
    print("Confusion matrix:\n", cm)

    # Raw counts
    plot_confusion(cm, class_names, acc, auc, out_raw, normalize=False)

    # Row-normalized
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums
    plot_confusion(cm_norm, class_names, acc, auc, out_norm, normalize=True)


if __name__ == "__main__":
    main()
