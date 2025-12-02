#!/usr/bin/env python3
"""
ImageNet SReC Runner

This script runs SReC detection on ImageNet datasets, handling the nested directory
structure and generating the required file lists for each dataset.

Usage:
    python imagenet_srec_runner.py --max-images 100
"""

import os
import subprocess
import tempfile
from pathlib import Path
import json
import argparse

# Dataset paths - same as used in RIGID
IMAGENET_DATASETS = {
    "val": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/imagenet256/imagenet256/val",
    "imagenet256-adm": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-ADM/Imagenet256-ADM",
    "imagenet256-admg": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-ADMG/Imagenet256-ADMG",
    "imagenet256-biggan": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-BigGAN/Imagenet256-BigGAN",
    "imagenet256-dit-xl-2": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-DiT-XL-2/Imagenet256-DiT-XL-2",
    "imagenet256-gigagan": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-GigaGAN/Imagenet256-GigaGAN",
    "imagenet256-ldm": "/mnt/hdd-data/vaidya/RIGID/gen_images/Generated_Images/ImageNet/extracted/Imagenet256-LDM/Imagenet256-LDM",
}

# SReC model and configuration
SREC_MODEL = "/mnt/ssd-data/vaidya/SReC/models/imagenet64.pth"
SREC_SCRIPT = "/mnt/ssd-data/vaidya/SReC/srec_detector.py"
BASE_OUTPUT_DIR = "/mnt/ssd-data/vaidya/combined_pipeline/results/SReC/ImageNet"


def get_image_filenames(dir_path: str, max_images: int = None) -> list:
    """
    Get list of image filenames from directory, handling nested structure.
    Same logic as RIGID's get_image_filenames but returns relative paths.
    """
    dir_path = Path(dir_path)
    extensions = ['.png', '.jpg', '.jpeg']

    found_files = []
    for ext in extensions:
        # Use rglob for recursive search to handle nested directories
        found_files.extend([str(f.relative_to(dir_path)) for f in dir_path.rglob(f'*{ext}')])

    if max_images and len(found_files) > max_images:
        found_files = found_files[:max_images]

    print(f"Found {len(found_files)} images in {dir_path}")
    return found_files


def create_file_list(filenames: list, output_dir: str, dataset_name: str) -> str:
    """Create permanent file with list of image filenames for SReC."""
    file_list_path = os.path.join(output_dir, f"{dataset_name}_image_list.txt")
    with open(file_list_path, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")
    print(f"Created image list: {file_list_path}")
    return file_list_path


def run_srec_on_dataset(dataset_name: str, dataset_path: str, max_images: int):
    """Run SReC detection on a single dataset."""
    print(f"\n{'=' * 60}")
    print(f"Processing {dataset_name.upper()}")
    print(f"Path: {dataset_path}")
    print(f"{'=' * 60}")

    # Get image filenames
    filenames = get_image_filenames(dataset_path, max_images)
    if not filenames:
        print(f"No images found in {dataset_path}")
        return None

    # Create output directory
    output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name.upper())
    os.makedirs(output_dir, exist_ok=True)

    # Create permanent file list in the output directory
    file_list_path = create_file_list(filenames, output_dir, dataset_name)

    # Run SReC detector
    cmd = [
        "python", SREC_SCRIPT,
        "--path", dataset_path,
        "--file", file_list_path,
        "--resblocks", "3",
        "--n-feats", "64",
        "--scale", "3",
        "--load", SREC_MODEL,
        "--K", "10",
        "--crop", "0",
        "--log-likelihood",
        "--save-path", output_dir
    ]

    print(f"Running SReC on {len(filenames)} images...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SReC processing completed successfully!")
        print(f"Results saved to: {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running SReC on {dataset_name}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def summarize_results(output_dirs: dict):
    """Load and summarize D(l) results from all datasets."""
    print(f"\n{'=' * 60}")
    print("SREC RESULTS SUMMARY")
    print(f"{'=' * 60}")

    all_results = {}
    for dataset_name, output_dir in output_dirs.items():
        if output_dir is None:
            continue

        # Look for JSON results file
        json_files = list(Path(output_dir).glob("*_d0.json"))
        if not json_files:
            print(f"No results file found for {dataset_name}")
            continue

        json_file = json_files[0]
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)

            # Calculate statistics
            d_values = list(results.values())
            mean_d = sum(d_values) / len(d_values)
            min_d = min(d_values)
            max_d = max(d_values)

            all_results[dataset_name] = {
                'mean_d': mean_d,
                'min_d': min_d,
                'max_d': max_d,
                'count': len(d_values),
                'std_d': (sum((x - mean_d) ** 2 for x in d_values) / len(d_values)) ** 0.5
            }

            print(
                f"{dataset_name:20}: mean_D(l)={mean_d:8.4f}, std={all_results[dataset_name]['std_d']:6.4f}, count={len(d_values):4d}")

        except Exception as e:
            print(f"Error reading results for {dataset_name}: {e}")

    # Save summary
    summary_file = os.path.join(BASE_OUTPUT_DIR, "srec_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    return all_results


def main():
    """
    Run SReC detection on ImageNet datasets.

    This script processes the same ImageNet datasets used in RIGID:
    - Real ImageNet validation set
    - 6 AI generators: ADM, ADMG, BigGAN, DiT-XL-2, GigaGAN, LDM
    """
    parser = argparse.ArgumentParser(description="ImageNet SReC Detection Pipeline")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Maximum number of images to process per dataset")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated list of datasets to process, or 'all'")
    args = parser.parse_args()

    max_images = args.max_images
    datasets = args.datasets
    print("ImageNet SReC Detection Pipeline")
    print(f"Max images per dataset: {max_images}")
    print(f"SReC model: {SREC_MODEL}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")

    # Determine which datasets to process
    if datasets == "all":
        datasets_to_process = IMAGENET_DATASETS
    else:
        dataset_names = [name.strip() for name in datasets.split(",")]
        datasets_to_process = {name: IMAGENET_DATASETS[name] for name in dataset_names if name in IMAGENET_DATASETS}

    print(f"Processing {len(datasets_to_process)} datasets...")

    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Process each dataset
    output_dirs = {}
    for dataset_name, dataset_path in datasets_to_process.items():
        output_dir = run_srec_on_dataset(dataset_name, dataset_path, max_images)
        output_dirs[dataset_name] = output_dir

    # Summarize results
    summarize_results(output_dirs)

    print(f"\n{'=' * 60}")
    print("ImageNet SReC processing complete!")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()