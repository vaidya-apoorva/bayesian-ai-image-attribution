#!/usr/bin/env python3
"""
SReC Results JSON Storage

This script converts SReC ImageNet results into structured JSON format
for easy access and integration with other pipeline components.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def create_srec_results_json():
    """Create comprehensive JSON file with SReC ImageNet results."""

    # Your SReC ImageNet results
    srec_results = {
        "metadata": {
            "experiment_name": "SReC_ImageNet_FullExperiment",
            "date_processed": "2025-10-23",
            "total_datasets": 7,
            "images_per_dataset": 512,
            "total_images": 3584,
            "model_used": "imagenet64.pth",
            "processing_method": "CUDA-only with RGB conversion",
            "description": "Full SReC experiment on ImageNet validation and 6 AI generators"
        },

        "results": {
            # Real dataset
            "val": {
                "type": "real",
                "dataset_name": "ImageNet Validation",
                "mean_d_l": -0.0271,
                "std_d_l": 0.0339,
                "count": 512,
                "interpretation": "Negative D(l) indicates easier compression (expected for real images)",
                "detectability": "baseline"
            },

            # AI generators
            "imagenet256-adm": {
                "type": "ai",
                "dataset_name": "ImageNet256 ADM",
                "mean_d_l": 0.0126,
                "std_d_l": 0.0654,
                "count": 512,
                "interpretation": "Positive D(l) indicates harder compression than real images",
                "detectability": "medium"
            },

            "imagenet256-admg": {
                "type": "ai",
                "dataset_name": "ImageNet256 ADMG",
                "mean_d_l": 0.0095,
                "std_d_l": 0.0642,
                "count": 512,
                "interpretation": "Positive D(l) indicates harder compression than real images",
                "detectability": "medium"
            },

            "imagenet256-biggan": {
                "type": "ai",
                "dataset_name": "ImageNet256 BigGAN",
                "mean_d_l": -0.0426,
                "std_d_l": 0.0344,
                "count": 512,
                "interpretation": "Most negative D(l) - even easier to compress than real images",
                "detectability": "high"
            },

            "imagenet256-dit-xl-2": {
                "type": "ai",
                "dataset_name": "ImageNet256 DiT-XL-2",
                "mean_d_l": 0.0477,
                "std_d_l": 0.0477,
                "count": 512,
                "interpretation": "Highest positive D(l) - most difficult to compress",
                "detectability": "very_high"
            },

            "imagenet256-gigagan": {
                "type": "ai",
                "dataset_name": "ImageNet256 GigaGAN",
                "mean_d_l": -0.0281,
                "std_d_l": 0.0380,
                "count": 512,
                "interpretation": "Negative D(l) similar to real images",
                "detectability": "low"
            },

            "imagenet256-ldm": {
                "type": "ai",
                "dataset_name": "ImageNet256 LDM",
                "mean_d_l": -0.0053,
                "std_d_l": 0.0338,
                "count": 512,
                "interpretation": "Near-zero D(l) - compression similar to real images",
                "detectability": "low"
            }
        },

        "analysis": {
            "real_datasets": {
                "count": 1,
                "datasets": ["val"],
                "mean_d_l_range": [-0.0271, -0.0271],
                "avg_mean_d_l": -0.0271,
                "avg_std_d_l": 0.0339
            },

            "ai_datasets": {
                "count": 6,
                "datasets": ["imagenet256-adm", "imagenet256-admg", "imagenet256-biggan",
                             "imagenet256-dit-xl-2", "imagenet256-gigagan", "imagenet256-ldm"],
                "mean_d_l_range": [-0.0426, 0.0477],
                "avg_mean_d_l": -0.0010,  # Average across all AI generators
                "avg_std_d_l": 0.0472
            },

            "detectability_ranking": [
                {
                    "rank": 1,
                    "dataset": "imagenet256-dit-xl-2",
                    "mean_d_l": 0.0477,
                    "detection_reason": "Highest positive D(l) - most compressible artifacts"
                },
                {
                    "rank": 2,
                    "dataset": "imagenet256-biggan",
                    "mean_d_l": -0.0426,
                    "detection_reason": "Most negative D(l) - overly compressible (suspicious)"
                },
                {
                    "rank": 3,
                    "dataset": "imagenet256-adm",
                    "mean_d_l": 0.0126,
                    "detection_reason": "Positive D(l) with high variance"
                },
                {
                    "rank": 4,
                    "dataset": "imagenet256-admg",
                    "mean_d_l": 0.0095,
                    "detection_reason": "Positive D(l) with high variance"
                },
                {
                    "rank": 5,
                    "dataset": "imagenet256-gigagan",
                    "mean_d_l": -0.0281,
                    "detection_reason": "Negative D(l) similar to real but distinguishable"
                },
                {
                    "rank": 6,
                    "dataset": "imagenet256-ldm",
                    "mean_d_l": -0.0053,
                    "detection_reason": "Near-zero D(l) - hardest to distinguish from real"
                }
            ],

            "compression_patterns": {
                "easier_than_real": {
                    "datasets": ["imagenet256-biggan", "imagenet256-gigagan"],
                    "description": "These AI generators produce images that are even easier to compress than real images"
                },
                "harder_than_real": {
                    "datasets": ["imagenet256-adm", "imagenet256-admg", "imagenet256-dit-xl-2"],
                    "description": "These AI generators produce images that are harder to compress than real images"
                },
                "similar_to_real": {
                    "datasets": ["imagenet256-ldm"],
                    "description": "This generator produces images with compression characteristics very similar to real images"
                }
            }
        },

        "statistical_summary": {
            "all_datasets": {
                "d_l_range": [-0.0426, 0.0477],
                "d_l_span": 0.0903,
                "mean_separation": {
                    "real_vs_ai_avg": -0.0261,  # val mean - AI average
                    "max_separation": 0.0748,  # val vs dit-xl-2
                    "min_separation": -0.0218  # val vs ldm
                }
            },

            "variance_analysis": {
                "lowest_variance": {
                    "dataset": "imagenet256-ldm",
                    "std": 0.0338
                },
                "highest_variance": {
                    "dataset": "imagenet256-adm",
                    "std": 0.0654
                },
                "real_baseline_variance": {
                    "dataset": "val",
                    "std": 0.0339
                }
            }
        },

        "bayesian_integration": {
            "priors_available": True,
            "rigid_kde_integration": "Available - use rigid_per_generator_distributions.json",
            "srec_d_l_values": "Current results can serve as likelihood estimates",
            "next_steps": [
                "Train 1-class detectors for each dataset",
                "Generate 7x7 likelihood matrix",
                "Combine RIGID priors with SReC likelihoods",
                "Evaluate Bayesian framework performance"
            ]
        }
    }

    return srec_results


def save_srec_results():
    """Save SReC results to JSON files."""

    # Create results
    srec_data = create_srec_results_json()

    # Output directory
    output_dir = Path("/mnt/ssd-data/vaidya/combined_pipeline/results/SReC")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comprehensive results
    comprehensive_path = output_dir / "srec_imagenet_comprehensive_results.json"
    with open(comprehensive_path, 'w') as f:
        json.dump(srec_data, f, indent=2)

    print(f"Comprehensive SReC results saved: {comprehensive_path}")

    # Save simplified results for easy integration
    simplified_results = {
        "datasets": {},
        "summary": {
            "real_mean": -0.0271,
            "ai_means": {
                "imagenet256-adm": 0.0126,
                "imagenet256-admg": 0.0095,
                "imagenet256-biggan": -0.0426,
                "imagenet256-dit-xl-2": 0.0477,
                "imagenet256-gigagan": -0.0281,
                "imagenet256-ldm": -0.0053
            }
        }
    }

    # Add dataset-wise data
    for dataset_name, data in srec_data["results"].items():
        simplified_results["datasets"][dataset_name] = {
            "mean_d_l": data["mean_d_l"],
            "std_d_l": data["std_d_l"],
            "type": data["type"],
            "count": data["count"]
        }

    simplified_path = output_dir / "srec_imagenet_simple.json"
    with open(simplified_path, 'w') as f:
        json.dump(simplified_results, f, indent=2)

    print(f"Simplified SReC results saved: {simplified_path}")

    # Save just the D(l) values for Bayesian integration
    dl_values = {}
    for dataset_name, data in srec_data["results"].items():
        dl_values[dataset_name] = data["mean_d_l"]

    dl_path = output_dir / "srec_d_l_values.json"
    with open(dl_path, 'w') as f:
        json.dump(dl_values, f, indent=2)

    print(f"D(l) values for Bayesian integration saved: {dl_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SReC ImageNet Results Summary")
    print(f"{'=' * 60}")
    print(f"Real dataset (val): D(l) = {srec_data['results']['val']['mean_d_l']:.4f}")
    print(f"AI generators:")

    ai_results = [(name, data['mean_d_l']) for name, data in srec_data['results'].items()
                  if data['type'] == 'ai']
    ai_results.sort(key=lambda x: x[1], reverse=True)  # Sort by D(l) descending

    for name, dl_value in ai_results:
        print(f"  {name:25}: D(l) = {dl_value:8.4f}")

    print(f"\nFiles created:")
    print(f"  - Comprehensive: srec_imagenet_comprehensive_results.json")
    print(f"  - Simplified: srec_imagenet_simple.json")
    print(f"  - D(l) values: srec_d_l_values.json")
    print(f"{'=' * 60}")

    return comprehensive_path, simplified_path, dl_path


if __name__ == "__main__":
    save_srec_results()