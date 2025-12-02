#!/usr/bin/env python3
"""
Test RIGID with different noise intensities to see if we can improve discrimination.
"""
import sys

sys.path.append('/mnt/ssd-data/vaidya/RIGID')

from rigid_detector import detect_images
import json
import numpy as np


def test_noise_levels():
    """Test different noise intensities."""

    # Test datasets (small subset for quick testing)
    dataset_paths = {
        'coco': '/mnt/hdd-data/vaidya/dataset/coco',
        'dalle2': '/mnt/hdd-data/vaidya/dataset/dalle2'
    }

    real_datasets = ['coco']
    ai_datasets = ['dalle2']

    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

    print("Testing different noise intensities...")
    print("=" * 60)

    for noise in noise_levels:
        print(f"\nTesting noise intensity: {noise}")

        try:
            results = detect_images(
                dataset_paths=dataset_paths,
                real_datasets=real_datasets,
                ai_datasets=ai_datasets,
                output_dir=f'./test_noise_{noise}',
                noise_intensity=noise,
                max_images=50,  # Small subset for quick test
                batch_size=32
            )

            # Get similarity scores
            coco_scores = list(results['coco'].values())
            dalle2_scores = list(results['dalle2'].values())

            # Calculate basic stats
            coco_mean = np.mean(coco_scores)
            dalle2_mean = np.mean(dalle2_scores)
            difference = abs(coco_mean - dalle2_mean)

            print(f"  COCO (Real):  mean={coco_mean:.4f}")
            print(f"  DALLE2 (AI):  mean={dalle2_mean:.4f}")
            print(f"  Difference:   {difference:.4f}")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_noise_levels()