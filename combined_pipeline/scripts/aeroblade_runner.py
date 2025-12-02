#!/usr/bin/env python3
"""
Aeroblade Pipeline Runner

Script to execute Aeroblade distance computation and organize outputs.
This processes multiple datasets and computes LPIPS-based reconstruction distances
using various diffusion model autoencoders.

Usage:
    # Single dataset
    python aeroblade_runner.py --images /path/to/images --output /path/to/output

    # Multiple datasets
    python aeroblade_runner.py --images-list /path/to/dataset1 /path/to/dataset2 --output /path/to/output
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
import time
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AerobladeRunner:
    """Runner class for executing Aeroblade distance computation."""

    def __init__(self, output_base="/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/priors/AEROBLADE"):
        """Initialize the runner.

        Args:
            output_base: Base directory for all Aeroblade outputs
        """
        self.aeroblade_dir = Path("/mnt/ssd-data/vaidya/aeroblade")
        self.output_base = Path(output_base)
        self.repo_ids = [
            "CompVis/stable-diffusion-v1-1",
            "stabilityai/stable-diffusion-2-base",
            "kandinsky-community/kandinsky-2-1",
        ]
        self.distance_metrics = ["lpips_vgg_2"]

    def run_aeroblade(self, image_dir, output_dir, amount=None):
        """Run Aeroblade distance computation."""
        logger.info("Running Aeroblade distance computation...")

        # Import aeroblade modules
        sys.path.insert(0, str(self.aeroblade_dir / "src"))

        try:
            from aeroblade.high_level_funcs import compute_distances

            # Convert to Path
            image_dir = Path(image_dir)
            output_dir = Path(output_dir)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Compute distances
            logger.info(f"Computing distances for images in: {image_dir}")
            logger.info(f"Processing images at original size (batch_size=1)")
            logger.info(f"Repo IDs: {self.repo_ids}")
            logger.info(f"Distance metrics: {self.distance_metrics}")

            distances = compute_distances(
                dirs=[image_dir],
                transforms=["clean"],  # Use clean - no preprocessing
                repo_ids=self.repo_ids,
                distance_metrics=self.distance_metrics,
                amount=amount,
                reconstruction_root=output_dir / "reconstructions",
                seed=1,
                batch_size=1,  # Use batch_size=1 for variable sizes
                num_workers=0,  # Avoid worker issues
                compute_max=True,
            )

            # Filter to keep only 'max' repo_id rows (most discriminative)
            distances_max = distances[distances['repo_id'] == 'max'].copy()
            logger.info(f"Filtered to max distances only: {len(distances_max)} images")

            # Save distances (only max values)
            distances_file = output_dir / "distances.parquet"
            distances_max.to_parquet(distances_file)
            logger.info(f"Saved max distances to: {distances_file}")

            # Also save as CSV for easier inspection
            distances_csv = output_dir / "distances.csv"
            distances_max.to_csv(distances_csv, index=False)
            logger.info(f"Saved max distances CSV to: {distances_csv}")

            return True

        except Exception as e:
            logger.error(f"Error running Aeroblade: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_dataset(self, image_dir, amount=None):
        """Process a single dataset through Aeroblade pipeline."""
        start_time = time.time()

        # Get dataset name
        dataset_name = Path(image_dir).name
        logger.info(f"Processing dataset: {dataset_name}")

        # Create output directory
        output_dir = self.output_base / dataset_name.upper()
        os.makedirs(output_dir, exist_ok=True)

        # Run Aeroblade
        success = self.run_aeroblade(image_dir, output_dir, amount=amount)

        if not success:
            logger.error(f"Failed to process {dataset_name}")
            return False

        # Create JSON format for integration with Bayesian pipeline
        success = self.create_json_output(output_dir, dataset_name)
        if not success:
            logger.warning(f"Failed to create JSON output for {dataset_name}")

        total_time = time.time() - start_time
        logger.info(f"Completed {dataset_name} in {total_time:.2f} seconds")
        return True

    def create_json_output(self, output_dir, dataset_name):
        """Convert Aeroblade distances to JSON format compatible with Bayesian pipeline."""
        try:
            output_dir = Path(output_dir)
            distances_file = output_dir / "distances.parquet"

            if not distances_file.exists():
                logger.error(f"Distances file not found: {distances_file}")
                return False

            # Read distances
            df = pd.read_parquet(distances_file)

            # For each image, get the max distance (most discriminative)
            max_distances = df[df['repo_id'] == 'max'].copy()

            # Create JSON format: {filename: distance_value}
            json_data = {}
            for _, row in max_distances.iterrows():
                filename = row['file']
                distance = float(row['distance'])
                json_data[filename] = distance

            # Save JSON
            json_file = output_dir / f"{dataset_name}_aeroblade.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)

            logger.info(f"Created JSON output: {json_file} ({len(json_data)} images)")

            # Also copy to centralized location
            json_target_dir = Path("/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/priors/AEROBLADE")
            os.makedirs(json_target_dir, exist_ok=True)
            json_target = json_target_dir / f"{dataset_name}_aeroblade.json"
            shutil.copy2(json_file, json_target)
            logger.info(f"Copied JSON to: {json_target}")

            return True

        except Exception as e:
            logger.error(f"Error creating JSON output: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for the Aeroblade pipeline runner."""
    parser = argparse.ArgumentParser(description="Aeroblade Pipeline Runner")

    # Allow multiple input paths
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Single directory containing images to analyze")
    group.add_argument("--images-list", nargs='+', help="Multiple directories containing images to analyze")

    parser.add_argument("--output",
                        default="/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/priors/AEROBLADE",
                        help="Base output directory (default: /mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/priors/AEROBLADE)")
    parser.add_argument("--amount", type=int, help="Limit number of images per dataset (default: all)")
    parser.add_argument("--repo-ids", nargs='+',
                        default=["CompVis/stable-diffusion-v1-1",
                                 "stabilityai/stable-diffusion-2-base",
                                 "kandinsky-community/kandinsky-2-1"],
                        help="Diffusion model repo IDs to use")
    parser.add_argument("--distance-metrics", nargs='+', default=["lpips_vgg_2"],
                        help="Distance metrics to compute")

    args = parser.parse_args()

    # Determine input directories to process
    if args.images:
        input_dirs = [args.images]
    else:
        input_dirs = args.images_list

    # Initialize runner
    runner = AerobladeRunner(output_base=args.output)
    runner.repo_ids = args.repo_ids
    runner.distance_metrics = args.distance_metrics

    # Process each input directory
    all_success = True
    results_summary = {}

    for input_dir in input_dirs:
        # Validate input directory
        if not Path(input_dir).exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            all_success = False
            continue

        dataset_name = Path(input_dir).name
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"Input: {input_dir}")
        logger.info(f"{'=' * 60}")

        # Process dataset
        success = runner.process_dataset(input_dir, amount=args.amount)

        if success:
            output_path = runner.output_base / dataset_name.upper()
            results_summary[dataset_name] = str(output_path)
            logger.info(f"✓ Successfully processed {dataset_name}")
        else:
            all_success = False
            logger.error(f"✗ Failed to process {dataset_name}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    for dataset, output_path in results_summary.items():
        logger.info(f"✓ {dataset}: {output_path}")

    if not all_success:
        logger.warning("Some datasets failed to process. Check logs above.")
    else:
        logger.info("\n✓ All datasets processed successfully!")
        logger.info(
            f"Aeroblade JSON files saved to: /mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/priors/AEROBLADE/")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
