#!/usr/bin/env python3
"""
RIGID Results JSON Storage

This script converts RIGID ImageNet results into structured JSON format
for easy access and integration with other pipeline components.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def create_rigid_results_json():
    """Create comprehensive JSON file with RIGID ImageNet results."""

    # Your RIGID ImageNet results from the log output
    rigid_results = {
        "metadata": {
            "experiment_name": "RIGID_ImageNet_FullExperiment",
            "date_processed": "2025-10-21",
            "total_datasets": 7,
            "images_per_dataset": 512,
            "total_images": 3584,
            "model_used": "DINOv2 ViT-L/14",
            "processing_method": "Noise perturbation with similarity calculation",
            "description": "Full RIGID experiment on ImageNet validation and 6 AI generators",
            "execution_time": "162.90 seconds",
            "overall_metrics": {
                "AUROC": 0.8676,
                "FPR95": 0.4141,
                "Average_Precision": 0.4761
            }
        },

        "results": {
            # Real dataset
            "val": {
                "type": "real",
                "dataset_name": "ImageNet Validation",
                "mean_similarity": 0.9934,
                "count": 512,
                "interpretation": "Highest similarity to noise - typical for real images",
                "detectability": "baseline",
                "source_path": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/imagenet256/imagenet256/val",
                "total_available": 50000
            },

            # AI generators
            "imagenet256-adm": {
                "type": "ai",
                "dataset_name": "ImageNet256 ADM",
                "mean_similarity": 0.9638,
                "count": 512,
                "interpretation": "Lower similarity than real - detectable AI artifacts",
                "detectability": "medium",
                "source_path": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-ADM/Imagenet256-ADM",
                "total_available": 50000
            },

            "imagenet256-admg": {
                "type": "ai",
                "dataset_name": "ImageNet256 ADMG",
                "mean_similarity": 0.9726,
                "count": 512,
                "interpretation": "Moderate similarity - some AI artifacts present",
                "detectability": "medium",
                "source_path": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-ADMG/Imagenet256-ADMG",
                "total_available": 50000
            },

            "imagenet256-biggan": {
                "type": "ai",
                "dataset_name": "ImageNet256 BigGAN",
                "mean_similarity": 0.9510,
                "count": 512,
                "interpretation": "Lowest similarity - most detectable AI artifacts",
                "detectability": "very_high",
                "source_path": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-BigGAN/Imagenet256-BigGAN",
                "total_available": 50000
            },

            "imagenet256-dit-xl-2": {
                "type": "ai",
                "dataset_name": "ImageNet256 DiT-XL-2",
                "mean_similarity": 0.9821,
                "count": 512,
                "interpretation": "High similarity - closer to real images than other generators",
                "detectability": "low",
                "source_path": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-DiT-XL-2/Imagenet256-DiT-XL-2",
                "total_available": 100000
            },

            "imagenet256-gigagan": {
                "type": "ai",
                "dataset_name": "ImageNet256 GigaGAN",
                "mean_similarity": 0.9628,
                "count": 512,
                "interpretation": "Lower similarity - detectable AI patterns",
                "detectability": "medium",
                "source_path": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-GigaGAN/Imagenet256-GigaGAN",
                "total_available": 100000
            },

            "imagenet256-ldm": {
                "type": "ai",
                "dataset_name": "ImageNet256 LDM",
                "mean_similarity": 0.9717,
                "count": 512,
                "interpretation": "Moderate similarity - some distinguishable patterns",
                "detectability": "medium",
                "source_path": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-LDM/Imagenet256-LDM",
                "total_available": 100000
            }
        },

        "analysis": {
            "real_datasets": {
                "count": 1,
                "datasets": ["val"],
                "similarity_range": [0.9934, 0.9934],
                "avg_similarity": 0.9934
            },

            "ai_datasets": {
                "count": 6,
                "datasets": ["imagenet256-adm", "imagenet256-admg", "imagenet256-biggan",
                             "imagenet256-dit-xl-2", "imagenet256-gigagan", "imagenet256-ldm"],
                "similarity_range": [0.9510, 0.9821],
                "avg_similarity": 0.9686,  # Average across all AI generators
                "similarity_gap_to_real": 0.0248  # val - AI average
            },

            "detectability_ranking": [
                {
                    "rank": 1,
                    "dataset": "imagenet256-biggan",
                    "similarity": 0.9510,
                    "gap_from_real": 0.0424,
                    "detection_reason": "Lowest similarity - most obvious AI artifacts"
                },
                {
                    "rank": 2,
                    "dataset": "imagenet256-gigagan",
                    "similarity": 0.9628,
                    "gap_from_real": 0.0306,
                    "detection_reason": "Clear similarity gap from real images"
                },
                {
                    "rank": 3,
                    "dataset": "imagenet256-adm",
                    "similarity": 0.9638,
                    "gap_from_real": 0.0296,
                    "detection_reason": "Significant similarity reduction"
                },
                {
                    "rank": 4,
                    "dataset": "imagenet256-ldm",
                    "similarity": 0.9717,
                    "gap_from_real": 0.0217,
                    "detection_reason": "Moderate similarity gap"
                },
                {
                    "rank": 5,
                    "dataset": "imagenet256-admg",
                    "similarity": 0.9726,
                    "gap_from_real": 0.0208,
                    "detection_reason": "Small but detectable similarity gap"
                },
                {
                    "rank": 6,
                    "dataset": "imagenet256-dit-xl-2",
                    "similarity": 0.9821,
                    "gap_from_real": 0.0113,
                    "detection_reason": "Closest to real images - hardest to detect"
                }
            ],

            "generator_performance": {
                "most_realistic": {
                    "dataset": "imagenet256-dit-xl-2",
                    "similarity": 0.9821,
                    "description": "Closest to real image noise patterns"
                },
                "least_realistic": {
                    "dataset": "imagenet256-biggan",
                    "similarity": 0.9510,
                    "description": "Most distinguishable from real images"
                },
                "similarity_spread": 0.0311  # dit-xl-2 - biggan
            }
        },

        "statistical_summary": {
            "all_datasets": {
                "similarity_range": [0.9510, 0.9934],
                "similarity_span": 0.0424,
                "real_vs_ai_separation": {
                    "max_gap": 0.0424,  # val vs biggan
                    "min_gap": 0.0113,  # val vs dit-xl-2
                    "avg_gap": 0.0248  # val vs AI average
                }
            },

            "detection_performance": {
                "auroc": 0.8676,
                "fpr95": 0.4141,
                "average_precision": 0.4761,
                "real_images": 512,
                "ai_images": 3072,
                "total_images": 3584
            }
        },

        "comparison_with_srec": {
            "rigid_advantages": [
                "Single forward pass - much faster than SReC",
                "No compression artifacts to consider",
                "Direct similarity measurement approach"
            ],
            "rigid_patterns": {
                "real_baseline": 0.9934,
                "ai_range": [0.9510, 0.9821],
                "clear_separation": "All AI generators show lower similarity than real"
            },
            "complementary_to_srec": "RIGID focuses on noise patterns, SReC on compression patterns"
        },

        "bayesian_integration": {
            "rigid_as_prior": "High-quality prior probabilities from KDE distributions",
            "rigid_similarity_values": "Current results provide generator-specific similarity baselines",
            "kde_distributions_available": "rigid_per_generator_distributions.json provides full distributions",
            "next_steps": [
                "Use RIGID similarities as Bayesian priors",
                "Combine with SReC D(l) values as likelihoods",
                "Train 1-class detectors for additional evidence",
                "Evaluate multi-modal Bayesian framework"
            ]
        }
    }

    return rigid_results


def save_rigid_results():
    """Save RIGID results to JSON files."""

    # Create results
    rigid_data = create_rigid_results_json()

    # Output directory
    output_dir = Path("/mnt/ssd-data/vaidya/combined_pipeline/results/RIGID")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comprehensive results
    comprehensive_path = output_dir / "rigid_imagenet_comprehensive_results.json"
    with open(comprehensive_path, 'w') as f:
        json.dump(rigid_data, f, indent=2)

    print(f"Comprehensive RIGID results saved: {comprehensive_path}")

    # Save simplified results for easy integration
    simplified_results = {
        "datasets": {},
        "summary": {
            "real_similarity": 0.9934,
            "ai_similarities": {
                "imagenet256-adm": 0.9638,
                "imagenet256-admg": 0.9726,
                "imagenet256-biggan": 0.9510,
                "imagenet256-dit-xl-2": 0.9821,
                "imagenet256-gigagan": 0.9628,
                "imagenet256-ldm": 0.9717
            },
            "metrics": {
                "auroc": 0.8676,
                "fpr95": 0.4141,
                "average_precision": 0.4761
            }
        }
    }

    # Add dataset-wise data
    for dataset_name, data in rigid_data["results"].items():
        simplified_results["datasets"][dataset_name] = {
            "similarity": data["mean_similarity"],
            "type": data["type"],
            "count": data["count"],
            "detectability": data["detectability"]
        }

    simplified_path = output_dir / "rigid_imagenet_simple.json"
    with open(simplified_path, 'w') as f:
        json.dump(simplified_results, f, indent=2)

    print(f"Simplified RIGID results saved: {simplified_path}")

    # Save just the similarity values for Bayesian integration
    similarity_values = {}
    for dataset_name, data in rigid_data["results"].items():
        similarity_values[dataset_name] = data["mean_similarity"]

    similarity_path = output_dir / "rigid_similarity_values.json"
    with open(similarity_path, 'w') as f:
        json.dump(similarity_values, f, indent=2)

    print(f"Similarity values for Bayesian integration saved: {similarity_path}")

    # Save comparison data (RIGID vs SReC)
    comparison_data = {
        "rigid_similarities": {dataset: data["mean_similarity"]
                               for dataset, data in rigid_data["results"].items()},
        "srec_d_l_values": {
            "val": -0.0271,
            "imagenet256-adm": 0.0126,
            "imagenet256-admg": 0.0095,
            "imagenet256-biggan": -0.0426,
            "imagenet256-dit-xl-2": 0.0477,
            "imagenet256-gigagan": -0.0281,
            "imagenet256-ldm": -0.0053
        },
        "correlation_analysis": {
            "note": "RIGID measures noise similarity (higher = more real-like)",
            "srec_note": "SReC measures compression difficulty (negative = easier compression)",
            "complementary": "Different detection mechanisms - suitable for Bayesian combination"
        }
    }

    comparison_path = output_dir / "rigid_vs_srec_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"RIGID vs SReC comparison saved: {comparison_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("RIGID ImageNet Results Summary")
    print(f"{'=' * 60}")
    print(f"Real dataset (val): Similarity = {rigid_data['results']['val']['mean_similarity']:.4f}")
    print(f"AI generators:")

    ai_results = [(name, data['mean_similarity']) for name, data in rigid_data['results'].items()
                  if data['type'] == 'ai']
    ai_results.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity descending

    for name, similarity in ai_results:
        gap = rigid_data['results']['val']['mean_similarity'] - similarity
        print(f"  {name:25}: Similarity = {similarity:.4f} (gap: {gap:.4f})")

    print(f"\nOverall Performance:")
    print(f"  AUROC: {rigid_data['metadata']['overall_metrics']['AUROC']:.4f}")
    print(f"  FPR95: {rigid_data['metadata']['overall_metrics']['FPR95']:.4f}")
    print(f"  AP:    {rigid_data['metadata']['overall_metrics']['Average_Precision']:.4f}")

    print(f"\nFiles created:")
    print(f"  - Comprehensive: rigid_imagenet_comprehensive_results.json")
    print(f"  - Simplified: rigid_imagenet_simple.json")
    print(f"  - Similarity values: rigid_similarity_values.json")
    print(f"  - RIGID vs SReC: rigid_vs_srec_comparison.json")
    print(f"{'=' * 60}")

    return comprehensive_path, simplified_path, similarity_path, comparison_path


if __name__ == "__main__":
    save_rigid_results()