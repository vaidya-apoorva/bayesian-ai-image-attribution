#!/usr/bin/env python3
"""
Test optimal noise levels for different AI generators.
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


def test_per_generator_noise():
    """Test different noise levels for each AI generator."""

    # Test different AI generators
    ai_generators = ['dalle2', 'dalle3', 'sdxl', 'stable_diffusion_1_5']
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15]

    print("Testing optimal noise levels per AI generator...")
    print("=" * 80)

    results_summary = {}

    for ai_gen in ai_generators:
        print(f"\nTesting {ai_gen.upper()}:")
        print("-" * 40)

        dataset_paths = {
            'coco': '/mnt/hdd-data/vaidya/dataset/coco',
            ai_gen: f'/mnt/hdd-data/vaidya/dataset/{ai_gen}'
        }

        best_auroc = 0
        best_noise = 0

        for noise in noise_levels:
            try:
                results = detect_images(
                    dataset_paths=dataset_paths,
                    real_datasets=['coco'],
                    ai_datasets=[ai_gen],
                    output_dir=f'./test_{ai_gen}_noise_{noise}',
                    noise_intensity=noise,
                    max_images=100,  # Reasonable subset
                    batch_size=32
                )

                # Calculate AUROC
                coco_scores = np.array(list(results['coco'].values()))
                ai_scores = np.array(list(results[ai_gen].values()))
                auroc = calculate_auroc(coco_scores, ai_scores)

                # Track best performance
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_noise = noise

                status = "🎯" if auroc > 0.7 else "✅" if auroc > 0.6 else "⚠️" if auroc > 0.55 else "❌"
                print(f"  Noise {noise:0.3f}: AUROC {auroc:.4f} {status}")

            except Exception as e:
                print(f"  Noise {noise:0.3f}: ERROR - {e}")

        results_summary[ai_gen] = {'best_noise': best_noise, 'best_auroc': best_auroc}
        print(f"  🏆 BEST: noise={best_noise}, AUROC={best_auroc:.4f}")

    print("\n" + "=" * 80)
    print("OPTIMAL NOISE LEVELS SUMMARY")
    print("=" * 80)

    for ai_gen, result in results_summary.items():
        quality = "🎯 Excellent" if result['best_auroc'] > 0.7 else "✅ Good" if result[
                                                                                   'best_auroc'] > 0.6 else "⚠️ Okay" if \
        result['best_auroc'] > 0.55 else "❌ Poor"
        print(
            f"{ai_gen.upper():<20} | Noise: {result['best_noise']:0.3f} | AUROC: {result['best_auroc']:.4f} {quality}")


if __name__ == "__main__":
    test_per_generator_noise()