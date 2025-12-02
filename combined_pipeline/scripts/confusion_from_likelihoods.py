#!/usr/bin/env python3
"""
confusion_from_likelihoods.py

Compute confusion matrices using only the 'likelihoods' dict in your bayesian_*.json files.
Prediction = argmax(likelihoods).

Usage:
  python confusion_from_likelihoods.py \
    --files /mnt/ssd-data/vaidya/gi_conference_results/bayesian_results_with_new_srec_weights/bayesian_pipeline_results/bayesian_srec_results.json \
    --output /mnt/ssd-data/vaidya/gi_conference_results/detectors_only_likelihoods_analysis
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys

# Canonical order - same as your analyze script
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
    "Firefly", "SD-XL", "DALL.E3", "Midj.-V6",
    "DALL.E2", "SD-1.5", "Midj.-V5", "Real"
]
DISPLAY_TO_CANON = {
    "Firefly":"firefly","SD-XL":"sdxl","DALL.E3":"dall-e3","Midj.-V6":"midjourneyV6",
    "DALL.E2":"dall-e2","SD-1.5":"stable_diffusion_1-5","Midj.-V5":"midjourneyV5","Real":"coco"
}
CANON_TO_DISPLAY = {v:k for k,v in DISPLAY_TO_CANON.items()}
REAL_TAGS = {"coco","imagenet","laion","real"}

def canonicalize_label(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s_l = s.lower()
    alias = {
        "dalle2":"dall-e2","dalle-2":"dall-e2","dall.e2":"dall-e2","dall-e-2":"dall-e2",
        "dalle3":"dall-e3","dalle-3":"dall-e3","dall.e3":"dall-e3","dall-e-3":"dall-e3",
        "midjourneyv5":"midjourneyV5","midj.-v5":"midjourneyV5","mjv5":"midjourneyV5","midjourney-v5":"midjourneyV5",
        "midjourneyv6":"midjourneyV6","midj.-v6":"midjourneyV6","mjv6":"midjourneyV6","midjourney-v6":"midjourneyV6",
        "sdxl":"sdxl","sd-xl":"sdxl","stable-diffusion-xl":"sdxl",
        "stable_diffusion_1-5":"stable_diffusion_1-5","stable-diffusion-1-5":"stable_diffusion_1-5","sd-1.5":"stable_diffusion_1-5","sd15":"stable_diffusion_1-5",
        "firefly":"firefly","coco":"coco","real":"coco","imagenet":"coco","laion":"coco"
    }
    for k,v in alias.items():
        if k in s_l:
            return v
    for cid in CANON_IDS:
        if s_l == cid.lower():
            return cid
    for k,v in alias.items():
        if k in s_l:
            return v
    if any(t in s_l for t in REAL_TAGS):
        return "coco"
    return None

def extract_true_generator_from_path(path_str):
    # mimic extract_true_generator behaviour: try filename, parents, dataset_512 pattern etc.
    if path_str is None:
        return None
    p = Path(path_str)
    # try direct canonicalization on full path string
    cand = canonicalize_label(str(p))
    if cand:
        return cand
    # try parents
    for parent in p.parents:
        cc = canonicalize_label(parent.name)
        if cc:
            return cc
    # try filename
    cc = canonicalize_label(p.name)
    if cc:
        return cc
    # dataset_512 style
    parts = [x.lower() for x in p.parts]
    if "dataset_512" in parts:
        try:
            idx = parts.index("dataset_512")
            cc = canonicalize_label(parts[idx+1])
            if cc:
                return cc
        except Exception:
            pass
    return None

def load_records(file_path: Path):
    try:
        data = json.loads(file_path.read_text())
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}", file=sys.stderr)
        return []
    # file expected to be a list of dicts
    if isinstance(data, dict):
        # maybe wrapped
        for k in ("results","items","data","predictions"):
            if k in data and isinstance(data[k], list):
                return data[k]
        return [data]
    if isinstance(data, list):
        return data
    return []

def predict_from_likelihoods(item):
    L = item.get("likelihoods") or item.get("likelihood") or item.get("likelihoods_dict")
    if not isinstance(L, dict):
        return None
    # canonicalize keys and pick argmax
    canon_scores = {}
    for k,v in L.items():
        ck = canonicalize_label(k)
        if ck and ck in CANON_IDS:
            try:
                canon_scores[ck] = float(v)
            except Exception:
                pass
    if not canon_scores:
        return None
    return max(canon_scores.items(), key=lambda x: x[1])[0]

def compute_and_save(y_true, y_pred, out_dir: Path, prefix: str):
    labels = CANON_IDS
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(all='ignore'):
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    acc = accuracy_score(y_true, y_pred) * 100.0
    report = classification_report(y_true, y_pred, labels=labels, target_names=[CANON_TO_DISPLAY[c] for c in labels], zero_division=0)

    # Save text summary
    with open(out_dir / f"{prefix}_summary.txt", "w") as f:
        f.write(f"Overall accuracy (likelihoods argmax): {acc:.2f}%\n\n")
        f.write("Classification report (per-class precision/recall/f1):\n")
        f.write(report + "\n\n")
        f.write("Raw confusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("Row-normalized confusion matrix:\n")
        f.write(np.array2string(cm_norm) + "\n")

    print(f"Saved textual summary to {out_dir / (prefix + '_summary.txt')}")

    # Plot raw confusion matrix
    fig, ax = plt.subplots(figsize=(9,8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f"{prefix} confusion matrix (counts)  acc={acc:.2f}%")
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([CANON_TO_DISPLAY.get(l,l) for l in labels], rotation=45, ha='right')
    ax.set_yticklabels([CANON_TO_DISPLAY.get(l,l) for l in labels])
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig_path = out_dir / f"{prefix}_confusion_raw.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{prefix}_confusion_raw.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(9,8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f"{prefix} confusion matrix (row-normalized)")
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([CANON_TO_DISPLAY.get(l,l) for l in labels], rotation=45, ha='right')
    ax.set_yticklabels([CANON_TO_DISPLAY.get(l,l) for l in labels])
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig_path2 = out_dir / f"{prefix}_confusion_norm.png"
    fig.savefig(fig_path2, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{prefix}_confusion_norm.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path2}")

    # Binary Real vs AI
    M = np.zeros((2,2), dtype=float)
    for t,p in zip(y_true, y_pred):
        ti = 1 if t == "coco" else 0
        pi = 1 if p == "coco" else 0
        M[ti,pi] += 1.0
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M_norm = M / row_sums
    np.save(out_dir / f"{prefix}_binary_real_ai_raw.npy", M)
    np.save(out_dir / f"{prefix}_binary_real_ai_norm.npy", M_norm)
    with open(out_dir / f"{prefix}_binary_real_ai.txt","w") as f:
        f.write("Raw (rows=true, cols=pred):\n"); f.write(np.array2string(M)+"\n\n")
        f.write("Row-normalized:\n"); f.write(np.array2string(M_norm)+"\n")
    print(f"Saved Real-vs-AI matrices to {out_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True, help="Paths to bayesian_*.json files to process")
    p.add_argument("--output", required=True, help="Output directory for confusion matrices and summaries")
    args = p.parse_args()

    out_dir = Path(args.output)
    all_records = []
    for f in args.files:
        pth = Path(f)
        if not pth.exists():
            print(f"File not found: {pth}", file=sys.stderr)
            continue
        recs = load_records(pth)
        print(f"Loaded {len(recs)} records from {pth.name}")
        all_records.append((pth.name, recs))

    # Process per-file and combined
    combined_true = []
    combined_pred = []
    for fname, recs in all_records:
        y_true = []
        y_pred = []
        for item in recs:
            true = extract_true_generator_from_path(item.get("image_path"))
            if not true:
                continue
            pred = predict_from_likelihoods(item)
            if not pred:
                continue
            y_true.append(true); y_pred.append(pred)
            combined_true.append(true); combined_pred.append(pred)
        if y_true:
            compute_and_save(y_true, y_pred, out_dir=out_dir / (Path(fname).stem), prefix=Path(fname).stem)
        else:
            print(f"No usable labeled items in {fname}")

    # Combined
    if combined_true:
        compute_and_save(combined_true, combined_pred, out_dir=out_dir / "combined", prefix="combined_likelihoods_argmax")
    else:
        print("No usable items across all files")

if __name__ == "__main__":
    main()
