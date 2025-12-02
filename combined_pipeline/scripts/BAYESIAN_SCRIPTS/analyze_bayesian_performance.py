"""
Analyze Bayesian Attribution Performance (Including Real Images)
- Uses EXACT display names and order:
  ["Firefly", "SD-XL", "DALL.E3", "Midj.-V6", "DALL.E2", "SD-1.5", "Midj.-V5", "Real"]
- Includes REAL images (coco) in attribution analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except Exception:
    try:
        plt.style.use('seaborn-darkgrid')
    except Exception:
        try:
            sns.set_theme(style='darkgrid')
        except Exception:
            plt.style.use('default')
try:
    sns.set_palette("husl")
except Exception:
    pass

# -----------------------------------------------------------------------------
# Canonical internal IDs and REQUIRED display names/order
# -----------------------------------------------------------------------------
CANON_IDS = [
    "firefly",
    "sdxl",
    "dall-e3",
    "midjourneyV6",
    "dall-e2",
    "stable_diffusion_1-5",
    "midjourneyV5",
    "coco",  # Added COCO as a legitimate class
]

DISPLAY_ORDER = [
    "Firefly",
    "SD-XL",
    "DALL.E3",
    "Midj.-V6",
    "DALL.E2",
    "SD-1.5",
    "Midj.-V5",
    "Real",  # Added Real as display name for COCO
]

DISPLAY_TO_CANON = {
    "Firefly": "firefly",
    "SD-XL": "sdxl",
    "DALL.E3": "dall-e3",
    "Midj.-V6": "midjourneyV6",
    "DALL.E2": "dall-e2",
    "SD-1.5": "stable_diffusion_1-5",
    "Midj.-V5": "midjourneyV5",
    "Real": "coco",  # Added mapping for Real -> coco
}
CANON_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_CANON.items()}

REAL_TAGS = {"coco", "imagenet", "laion", "real"}

ALIAS_TO_CANON = {
    "dalle2": "dall-e2",
    "dalle-2": "dall-e2",
    "dall-e-2": "dall-e2",
    "dall.e2": "dall-e2",
    "dalle3": "dall-e3",
    "dalle-3": "dall-e3",
    "dall-e-3": "dall-e3",
    "dall.e3": "dall-e3",
    "firefly": "firefly",
    "adobe_firefly": "firefly",
    "adobe-firefly": "firefly",
    "midjourneyv5": "midjourneyV5",
    "midjourney-v5": "midjourneyV5",
    "midj.-v5": "midjourneyV5",
    "mjv5": "midjourneyV5",
    "midjourneyv6": "midjourneyV6",
    "midjourney-v6": "midjourneyV6",
    "midj.-v6": "midjourneyV6",
    "mjv6": "midjourneyV6",
    "sdxl": "sdxl",
    "sd-xl": "sdxl",
    "stable-diffusion-xl": "sdxl",
    "stable_diffusion_1_5": "stable_diffusion_1-5",
    "stable-diffusion-1-5": "stable_diffusion_1-5",
    "stable_diffusion_1-5": "stable_diffusion_1-5",
    "sd15": "stable_diffusion_1-5",
    "sd-1.5": "stable_diffusion_1-5",
}

SUBSTRING_TOKENS = sorted(ALIAS_TO_CANON.keys(), key=len, reverse=True)

# -----------------------------------------------------------------------------
# Canonicalization helpers
# -----------------------------------------------------------------------------
def canonicalize_label(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s_l = s.lower()

    for cid in CANON_IDS:
        if s_l == cid.lower():
            return cid

    if s_l in ALIAS_TO_CANON:
        return ALIAS_TO_CANON[s_l]

    for token in SUBSTRING_TOKENS:
        if token in s_l:
            return ALIAS_TO_CANON[token]

    if any(tag in s_l for tag in REAL_TAGS):
        return "coco"  # Return coco instead of REAL to include in analysis

    return None


def extract_true_generator(image_path):
    p = Path(str(image_path))
    full = str(p).lower()

    cand = canonicalize_label(full)
    if cand:
        return cand

    for parent in p.parents:
        cand = canonicalize_label(parent.name)
        if cand:
            return cand

    cand = canonicalize_label(p.name)
    if cand:
        return cand

    parts = [x.lower() for x in p.parts]
    if "dataset_512" in parts:
        try:
            idx = parts.index("dataset_512")
            cand = canonicalize_label(parts[idx + 1])
            if cand:
                return cand
        except Exception:
            pass

    return None

# -----------------------------------------------------------------------------
# IO + parsing (including Real images)
# -----------------------------------------------------------------------------
def load_results(json_path: Path):
    data = json.loads(Path(json_path).read_text())

    results = []
    unmatched = 0
    matched_real = 0
    matched_counts = {cid: 0 for cid in CANON_IDS}

    for item in data:
        img_path = item.get("image_path", "")
        true_gen = extract_true_generator(img_path)

        if true_gen is None:
            unmatched += 1
            continue
        # Remove the skip for REAL images - now coco is included in CANON_IDS
        # if true_gen == "REAL":
        #     matched_real += 1
        #     continue

        pred_raw = item.get("predicted_generator")
        pred_gen = canonicalize_label(pred_raw)
        if pred_gen is None:
            continue
            continue

        if true_gen not in CANON_IDS or pred_gen not in CANON_IDS:
            continue

        matched_counts[true_gen] += 1

        post = item.get("posteriors", {})
        canon_post = {}
        if isinstance(post, dict):
            for k, v in post.items():
                ck = canonicalize_label(k)
                if ck in CANON_IDS:
                    try:
                        canon_post[ck] = float(v)
                    except Exception:
                        pass

        results.append({
            "image_path": img_path,
            "true_generator": true_gen,
            "predicted_generator": pred_gen,
            "confidence": float(item.get("confidence", 0.0)),
            "posteriors": canon_post,
            "correct": (true_gen == pred_gen),
        })

    print(f"Debug: matched_counts(including Real)={matched_counts}, matched_real_legacy={matched_real}, unmatched={unmatched}")
    return results

# -----------------------------------------------------------------------------
# Analyses
# -----------------------------------------------------------------------------
def analyze_performance(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc = (correct / total * 100.0) if total > 0 else 0.0

    confs = [r["confidence"] for r in results]
    confs_c = [r["confidence"] for r in results if r["correct"]]
    confs_w = [r["confidence"] for r in results if not r["correct"]]

    mean_conf = float(np.mean(confs)) if confs else 0.0
    median_conf = float(np.median(confs)) if confs else 0.0
    mean_conf_c = float(np.mean(confs_c)) if confs_c else 0.0
    mean_conf_w = float(np.mean(confs_w)) if confs_w else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,  # percent
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "mean_confidence_correct": mean_conf_c,
        "mean_confidence_wrong": mean_conf_w,
    }


def per_generator_analysis(results):
    stats = defaultdict(lambda: {"total": 0, "correct": 0, "confs": []})
    for r in results:
        g = r["true_generator"]
        stats[g]["total"] += 1
        stats[g]["correct"] += int(r["correct"])
        stats[g]["confs"].append(r["confidence"])

    per_gen = {}
    for cid in CANON_IDS:
        s = stats[cid]
        if s["total"] == 0:
            per_gen[cid] = {"count": 0, "accuracy": 0.0, "mean_confidence": 0.0, "median_confidence": 0.0}
        else:
            per_gen[cid] = {
                "count": s["total"],
                "accuracy": s["correct"] / s["total"] * 100.0,
                "mean_confidence": float(np.mean(s["confs"])),
                "median_confidence": float(np.median(s["confs"])),
            }
    return per_gen


def create_confusion_matrix(results):
    n = len(CANON_IDS)
    M = np.zeros((n, n), dtype=float)
    idx = {cid: i for i, cid in enumerate(CANON_IDS)}
    for r in results:
        ti = idx[r["true_generator"]]
        pi = idx[r["predicted_generator"]]
        M[ti, pi] += 1.0
    return M

# -----------------------------------------------------------------------------
# Plotting (uses requested display names/order)
# -----------------------------------------------------------------------------
def plot_accuracy_by_generator(per_gen, out_dir: Path):
    gens_disp = DISPLAY_ORDER[:]
    gens_canon = [DISPLAY_TO_CANON[g] for g in gens_disp]
    accs = [per_gen[c]["accuracy"] for c in gens_canon]
    counts = [per_gen[c]["count"] for c in gens_canon]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(gens_disp, accs, alpha=0.85)

    for b, c, a in zip(bars, counts, accs):
        ax.text(b.get_x() + b.get_width()/2., b.get_height(), f"{a:.1f}%\n(n={c})",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Generator", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Bayesian Attribution Accuracy by Generator", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 105])
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir.name  # e.g. "aeroblade" or "rigid"
    base_name = f"{prefix}_accuracy_by_generator"
    plt.tight_layout()
    plt.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{base_name}.pdf", bbox_inches="tight")
    plt.close()



# def plot_confidence_by_generator(per_gen, out_dir: Path):
#     gens_disp = DISPLAY_ORDER[:]
#     gens_canon = [DISPLAY_TO_CANON[g] for g in gens_disp]
#     confs = [per_gen[c]["mean_confidence"] for c in gens_canon]
#
#     fig, ax = plt.subplots(figsize=(10, 5))
#     bars = ax.bar(gens_disp, confs, alpha=0.85)
#
#     for b, c in zip(bars, confs):
#         ax.text(b.get_x() + b.get_width()/2., b.get_height(), f"{c:.3f}",
#                 ha="center", va="bottom", fontsize=9)
#
#     ax.set_xlabel("Generator", fontsize=11, fontweight="bold")
#     ax.set_ylabel("Mean Confidence", fontsize=11, fontweight="bold")
#     ax.set_title("Mean Prediction Confidence by Generator", fontsize=13, fontweight="bold")
#     ax.set_ylim([0, 1.05])
#     plt.xticks(rotation=45, ha="right", fontsize=9)
#     ax.grid(axis="y", alpha=0.3)
#
#     out_dir.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out_dir / "confidence_by_generator.png", dpi=300, bbox_inches="tight")
#     plt.close()
#

# def plot_confusion_matrices(M, acc_percent, out_dir: Path):
#     """Two-panel: raw counts + row-normalized, with EXACT display order and accuracy in title."""
#     row_sums = M.sum(axis=1, keepdims=True)
#     row_sums[row_sums == 0] = 1.0
#     M_norm = M / row_sums
#
#     labels = DISPLAY_ORDER[:]
#     acc_frac = (acc_percent / 100.0) if np.isfinite(acc_percent) else 0.0  # e.g., 0.749
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
#
#     sns.heatmap(M, annot=True, fmt=".0f", cmap="Blues",
#                 xticklabels=labels, yticklabels=labels,
#                 cbar_kws={"label": "Count"}, ax=ax1)
#     ax1.set_xlabel("Predicted Generator", fontsize=11, fontweight="bold")
#     ax1.set_ylabel("True Generator", fontsize=11, fontweight="bold")
#     ax1.set_title(f"Confusion Matrix (Counts)\nacc={acc_frac:.3f}", fontsize=13, fontweight="bold")
#     ax1.tick_params(axis="x", labelrotation=45)
#
#     sns.heatmap(M_norm, annot=True, fmt=".2f", cmap="Blues",
#                 xticklabels=labels, yticklabels=labels,
#                 cbar_kws={"label": "Proportion"}, ax=ax2, vmin=0, vmax=1)
#     ax2.set_xlabel("Predicted Generator", fontsize=11, fontweight="bold")
#     ax2.set_ylabel("True Generator", fontsize=11, fontweight="bold")
#     ax2.set_title(f"Confusion Matrix (Row-Normalized)\nacc={acc_frac:.3f}", fontsize=13, fontweight="bold")
#     ax2.tick_params(axis="x", labelrotation=45)
#
#     out_dir.mkdir(parents=True, exist_ok=True)
#     prefix = out_dir.name
#     plt.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=300, bbox_inches="tight")
#     plt.savefig(out_dir / f"{prefix}_confusion_matrix.pdf", bbox_inches="tight")
#     plt.close()
#


def plot_confusion_matrices(M, acc_percent, out_dir: Path):
    """Save raw-count and row-normalized confusion matrices as separate figures (both .png and .pdf)."""
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M_norm = M / row_sums

    labels = DISPLAY_ORDER[:]
    acc_frac = (acc_percent / 100.0) if np.isfinite(acc_percent) else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir.name

    # -------------------------------------------------------------------------
    # RAW COUNTS CONFUSION MATRIX
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(M, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Count"}, ax=ax)
    ax.set_xlabel("Predicted Generator", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Generator", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix (Counts)\nacc={acc_frac:.3f}", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45)

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_confusion_matrix_raw.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{prefix}_confusion_matrix_raw.pdf", bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # ROW-NORMALIZED CONFUSION MATRIX
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(M_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Proportion"}, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted Generator", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Generator", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix (Row-Normalized)\nacc={acc_frac:.3f}", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45)

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_confusion_matrix_norm.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{prefix}_confusion_matrix_norm.pdf", bbox_inches="tight")
    plt.close()

def create_confusion_matrix(results):
    n = len(CANON_IDS)
    M = np.zeros((n, n), dtype=float)
    idx = {cid: i for i, cid in enumerate(CANON_IDS)}
    for r in results:
        ti = idx[r["true_generator"]]
        pi = idx[r["predicted_generator"]]
        M[ti, pi] += 1.0
    return M


def plot_binary_confusion_real_vs_ai(results, out_dir: Path):
    """
    Build and save separate confusion matrices for Real vs AI:
    1. Raw counts
    2. Row-normalized proportions
    """
    # 0 = AI, 1 = Real
    M = np.zeros((2, 2), dtype=float)

    for r in results:
        true_is_real = (r["true_generator"] == "coco")
        pred_is_real = (r["predicted_generator"] == "coco")

        ti = 1 if true_is_real else 0
        pi = 1 if pred_is_real else 0
        M[ti, pi] += 1.0

    acc = (M[0, 0] + M[1, 1]) / M.sum() if M.sum() > 0 else 0.0
    acc_frac = float(acc)

    # Row-normalized matrix
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M_norm = M / row_sums

    labels = ["AI", "Real"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir.name

    # -------------------------------------------------------------------------
    # RAW COUNTS CONFUSION MATRIX
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(M, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Count"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    ax.set_ylabel("True", fontsize=11, fontweight="bold")
    ax.set_title(f"Real vs AI (Counts)\nacc={acc_frac:.3f}", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_binary_confusion_real_vs_ai_raw.png",
                dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{prefix}_binary_confusion_real_vs_ai_raw.pdf",
                bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # ROW-NORMALIZED CONFUSION MATRIX
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(M_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Proportion"}, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    ax.set_ylabel("True", fontsize=11, fontweight="bold")
    ax.set_title("Real vs AI (Row-Normalized)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_binary_confusion_real_vs_ai_norm.png",
                dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{prefix}_binary_confusion_real_vs_ai_norm.pdf",
                bbox_inches="tight")
    plt.close()

# def plot_confidence_distribution(results, out_dir: Path):
#     conf_c = [r["confidence"] for r in results if r["correct"]]
#     conf_w = [r["confidence"] for r in results if not r["correct"]]
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
#     ax1.hist(conf_c, bins=30, alpha=0.7, label="Correct", color="green", edgecolor="black")
#     ax1.hist(conf_w, bins=30, alpha=0.7, label="Incorrect", color="red", edgecolor="black")
#     ax1.set_xlabel("Confidence", fontsize=11, fontweight="bold")
#     ax1.set_ylabel("Frequency", fontsize=11, fontweight="bold")
#     ax1.set_title("Confidence Distribution", fontsize=13, fontweight="bold")
#     ax1.legend()
#     ax1.grid(alpha=0.3)
#
#     box = ax2.boxplot([conf_c, conf_w], labels=["Correct", "Incorrect"], patch_artist=True)
#     box["boxes"][0].set_facecolor("green"); box["boxes"][0].set_alpha(0.7)
#     box["boxes"][1].set_facecolor("red");   box["boxes"][1].set_alpha(0.7)
#     ax2.set_ylabel("Confidence", fontsize=11, fontweight="bold")
#     ax2.set_title("Confidence Box Plot", fontsize=13, fontweight="bold")
#     ax2.grid(axis="y", alpha=0.3)
#
#     out_dir.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight")
#     plt.close()
#

# def plot_accuracy_confidence_scatter(per_gen, out_dir: Path):
#     gens_disp = DISPLAY_ORDER[:]
#     gens_canon = [DISPLAY_TO_CANON[g] for g in gens_disp]
#     accs = [per_gen[c]["accuracy"] for c in gens_canon]
#     confs = [per_gen[c]["mean_confidence"] for c in gens_canon]
#     counts = [per_gen[c]["count"] for c in gens_canon]
#
#     fig, ax = plt.subplots(figsize=(9, 7))
#     ax.scatter(confs, accs, s=[max(10, c * 2) for c in counts],
#                alpha=0.6, c=range(len(gens_disp)), cmap="tab10", edgecolors="k")
#
#     for x, y, label in zip(confs, accs, gens_disp):
#         ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points",
#                     fontsize=9, fontweight="bold")
#
#     ax.set_xlabel("Mean Confidence", fontsize=11, fontweight="bold")
#     ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
#     ax.set_title("Accuracy vs Confidence by Generator", fontsize=13, fontweight="bold")
#     ax.grid(alpha=0.3)
#     ax.plot([0, 1], [0, 100], "k--", alpha=0.25, label="Perfect calibration")
#     ax.legend(loc="lower right", fontsize=9)
#
#     out_dir.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out_dir / "accuracy_vs_confidence.png", dpi=300, bbox_inches="tight")
#     plt.close()
#

def save_summary_report(overall, per_gen, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "performance_summary.txt"
    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BAYESIAN ATTRIBUTION PERFORMANCE (Including Real Images)\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total images analyzed: {overall['total']}\n")
        f.write(f"Correct predictions: {overall['correct']}\n")
        f.write(f"Overall accuracy: {overall['accuracy']:.2f}%\n")
        f.write(f"Mean confidence: {overall['mean_confidence']:.4f}\n")
        f.write(f"Median confidence: {overall['median_confidence']:.4f}\n")
        f.write(f"Mean confidence (correct): {overall['mean_confidence_correct']:.4f}\n")
        f.write(f"Mean confidence (incorrect): {overall['mean_confidence_wrong']:.4f}\n\n")

        f.write("PER-GENERATOR PERFORMANCE (Including Real Images):\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Generator':<12} {'Count':>7} {'Accuracy':>12} {'Mean Conf':>12}\n")
        f.write("-" * 80 + "\n")
        for disp in DISPLAY_ORDER:
            cid = DISPLAY_TO_CANON[disp]
            s = per_gen.get(cid, {"count": 0, "accuracy": 0.0, "mean_confidence": 0.0})
            f.write(f"{disp:<12} {s['count']:>7} {s['accuracy']:>10.2f}% {s['mean_confidence']:>11.4f}\n")
        f.write("\n" + "=" * 80 + "\n")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Bayesian Attribution Performance (Including Real Images; fixed display names)")
    parser.add_argument("--results", type=str, default="results/BAYESIAN_RESULTS/bayesian_attribution_results.json",
                        help="Path to the Bayesian attribution results JSON file")
    parser.add_argument("--output", type=str, default="results/bayesian_performance_analysis",
                        help="Output directory for analysis results")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.output)

    print(f"Results file: {results_path}")
    print(f"Output directory: {out_dir}")

    print("\nLoading results...")
    results = load_results(results_path)
    print(f"Loaded {len(results)} AI-only items")

    print("\nAnalyzing overall performance...")
    overall = analyze_performance(results)

    print("\nAnalyzing per-generator performance...")
    per_gen = per_generator_analysis(results)

    print("\nBuilding confusion matrix (including Real, fixed display order)...")
    M = create_confusion_matrix(results)

    print("\nPlotting...")
    plot_accuracy_by_generator(per_gen, out_dir);          print("  ✓ accuracy_by_generator")
    # plot_confidence_by_generator(per_gen, out_dir);        print("  ✓ confidence_by_generator")
    plot_confusion_matrices(M, overall['accuracy'], out_dir);  print("  ✓ confusion_matrix (raw+normalized)")
    # plot_confidence_distribution(results, out_dir);        print("  ✓ confidence_distribution")
    # plot_accuracy_confidence_scatter(per_gen, out_dir);    print("  ✓ accuracy_vs_confidence")
    plot_binary_confusion_real_vs_ai(results, out_dir); print("  ✓ binary_confusion_real_vs_ai")

    print("\nSaving summary...")
    save_summary_report(overall, per_gen, out_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE (including Real, fixed display names/order)")
    print("="*80)
    print(f"Overall Accuracy: {overall['accuracy']:.2f}%")
    print(f"Mean Confidence: {overall['mean_confidence']:.4f}")
    print(f"Outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
