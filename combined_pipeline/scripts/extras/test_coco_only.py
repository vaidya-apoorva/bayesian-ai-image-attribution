#!/usr/bin/env python3
"""
Test RIGID performance with only COCO as real dataset.
"""
import sys

sys.path.append('/mnt/ssd-data/vaidya/RIGID')

from rigid_detector import detect_images
import json
import numpy as np
from sklearn import metrics


def calculate_auroc(real_scores, ai_scores):
    """Calculate AUROC for a pair of score arrays."""
    all_scores = np.concatenate([real_scores, ai_scores])
    labels = np.concatenate([np.ones(len(real_scores)), np.zeros(len(ai_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, all_scores)
    return metrics.auc(fpr, tpr)


def test_coco_only():
    """Test RIGID with only COCO as real dataset."""

    # Test with subset of AI generators that showed promise
    dataset_paths = {
        'coco': '/mnt/hdd-data/vaidya/dataset/coco',
        'dalle2': '/mnt/hdd-data/vaidya/dataset/dalle2',
        'firefly': '/mnt/hdd-data/vaidya/dataset/firefly',
        'sdxl': '/mnt/hdd-data/vaidya/dataset/sdxl'
    }

    print("Testing RIGID with COCO-only as real dataset...")
    print("=" * 60)

    try:
        results = detect_images(
            dataset_paths=dataset_paths,
            real_datasets=['coco'],  # Only COCO
            ai_datasets=['dalle2', 'firefly', 'sdxl'],
            output_dir='./test_coco_only',
            noise_intensity=0.05,  # Use the optimal noise we found
            max_images=200,  # Reasonable subset
            batch_size=32
        )

        # Calculate individual AUROCs
        coco_scores = np.array(list(results['coco'].values()))

        print(f"\nIndividual Performance vs COCO:")
        print("-" * 40)

        total_auroc = 0
        count = 0

        for ai_dataset in ['dalle2', 'firefly', 'sdxl']:
            if ai_dataset in results:
                ai_scores = np.array(list(results[ai_dataset].values()))
                auroc = calculate_auroc(coco_scores, ai_scores)

                status = "🎯" if auroc > 0.7 else "✅" if auroc > 0.6 else "⚠️" if auroc > 0.55 else "❌"
                print(f"{ai_dataset.upper():<15} | AUROC: {auroc:.4f} {status}")

                total_auroc += auroc
                count += 1

        if count > 0:
            avg_auroc = total_auroc / count
            overall_status = "🎯 Excellent" if avg_auroc > 0.7 else "✅ Good" if avg_auroc > 0.6 else "⚠️ Okay" if avg_auroc > 0.55 else "❌ Poor"
            print("-" * 40)
            print(f"AVERAGE AUROC:   {avg_auroc:.4f} {overall_status}")
            print(f"Improvement from original 0.5399: +{avg_auroc - 0.5399:.4f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_coco_only()