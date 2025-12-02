#!/usr/bin/env python3
"""
Analyze individual dataset performance from RIGID results.
"""
import json
import numpy as np
from sklearn import metrics
import os


def calculate_auc_for_pair(real_scores, ai_scores):
    """Calculate AUROC for a real vs AI pair."""
    # Combine scores and create labels
    all_scores = np.concatenate([real_scores, ai_scores])
    # Real=1 (positive), AI=0 (negative)
    labels = np.concatenate([np.ones(len(real_scores)), np.zeros(len(ai_scores))])

    # Calculate AUROC
    fpr, tpr, _ = metrics.roc_curve(labels, all_scores)
    auroc = metrics.auc(fpr, tpr)

    return auroc


def analyze_rigid_results():
    """Analyze RIGID results per dataset pair."""

    results_dir = "/mnt/ssd-data/vaidya/combined_pipeline/results/RIGID"

    # Load real datasets
    real_datasets = ['COCO', 'RAISE']
    ai_datasets = ['DALLE2', 'DALLE3', 'FIREFLY', 'MIDJOURNEYV5', 'MIDJOURNEYV6', 'SDXL', 'STABLE_DIFFUSION_1_5']

    # Load all results
    all_results = {}

    for dataset in real_datasets + ai_datasets:
        json_file = os.path.join(results_dir, dataset, f"{dataset.lower()}_rigid_results.json")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            all_results[dataset] = list(data.values())
            print(f"Loaded {dataset}: {len(all_results[dataset])} images, mean={np.mean(all_results[dataset]):.4f}")
        except Exception as e:
            print(f"Error loading {dataset}: {e}")

    print("\n" + "=" * 80)
    print("INDIVIDUAL DATASET PAIR ANALYSIS")
    print("=" * 80)

    # Analyze each real vs AI pair
    for real_dataset in real_datasets:
        if real_dataset not in all_results:
            continue

        print(f"\n{real_dataset} vs AI datasets:")
        print("-" * 50)

        real_scores = np.array(all_results[real_dataset])

        for ai_dataset in ai_datasets:
            if ai_dataset not in all_results:
                continue

            ai_scores = np.array(all_results[ai_dataset])
            auroc = calculate_auc_for_pair(real_scores, ai_scores)

            # Calculate mean difference
            real_mean = np.mean(real_scores)
            ai_mean = np.mean(ai_scores)
            difference = real_mean - ai_mean

            status = "✅" if auroc > 0.6 else "⚠️" if auroc > 0.55 else "❌"
            direction = "Real>AI" if difference > 0 else "AI>Real"

            print(f"  {ai_dataset:<20} | AUROC: {auroc:.4f} {status} | Diff: {difference:+.4f} ({direction})")

    print("\n" + "=" * 80)
    print("SUMMARY BY AI GENERATOR")
    print("=" * 80)

    # Calculate average performance per AI dataset
    for ai_dataset in ai_datasets:
        if ai_dataset not in all_results:
            continue

        ai_scores = np.array(all_results[ai_dataset])
        aurocs = []

        for real_dataset in real_datasets:
            if real_dataset not in all_results:
                continue
            real_scores = np.array(all_results[real_dataset])
            auroc = calculate_auc_for_pair(real_scores, ai_scores)
            aurocs.append(auroc)

        if aurocs:
            avg_auroc = np.mean(aurocs)
            status = "✅ Good" if avg_auroc > 0.6 else "⚠️ Okay" if avg_auroc > 0.55 else "❌ Poor"
            print(f"{ai_dataset:<20} | Avg AUROC: {avg_auroc:.4f} {status}")


if __name__ == "__main__":
    analyze_rigid_results()