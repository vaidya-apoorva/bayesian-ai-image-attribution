#!/usr/bin/env python3
"""
RIGID Pipeline Runner

Script to execute RIGID detection using the improved rigid_detector.py.
This is a pipeline runner for testing RIGID (Training-free, model-agnostic detector) functionality.

RIGID computes per-image similarity between an image and its noise-perturbed version
in a vision backbone's feature space (default: DINOv2 ViT-L/14).

Usage:
    # Single dataset
    python rigid_runner.py --images /path/to/images

    # Multiple datasets
    python rigid_runner.py --images-list /path/to/dataset1 /path/to/dataset2

    # Specify which datasets are real vs AI-generated
    python rigid_runner.py --images-list /path/to/coco /path/to/dalle2 --real-datasets coco --ai-datasets dalle2
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import time

# Add project paths to system path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "RIGID"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RigidRunner:
    """Runner class for executing RIGID detection."""

    def __init__(self):
        """Initialize the runner."""
        self.rigid_dir = Path(__file__).parent.parent.parent / "RIGID"

        # Default dataset classifications - Only use COCO and ImageNet as real datasets
        self.default_real_datasets = ['coco', 'imagenet256']  # Removed 'raise' due to poor performance
        self.default_ai_datasets = ['dalle2', 'dalle3', 'firefly', 'midjourneyv5', 'midjourneyv6', 'sdxl',
                                    'stable_diffusion_1_5',
                                    'imagenet256-adm', 'imagenet256-admg', 'imagenet256-biggan', 'imagenet256-dit-xl-2',
                                    'imagenet256-gigagan', 'imagenet256-ldm']

    def create_image_list(self, image_dir, output_file):
        """Create image list file for verification."""
        try:
            image_dir = Path(image_dir)

            # Get list of all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = []

            for ext in image_extensions:
                image_files.extend(image_dir.rglob(f'*{ext}'))
                image_files.extend(image_dir.rglob(f'*{ext.upper()}'))

            # Write to file
            with open(output_file, 'w') as f:
                for img_file in sorted(image_files):
                    f.write(str(img_file) + '\n')

            logger.info(f"Created image list with {len(image_files)} images: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error creating image list: {e}")
            return False

    def run_rigid_detection(self, dataset_paths, real_datasets=None, ai_datasets=None,
                            output_dir='/mnt/ssd-data/vaidya/combined_pipeline/results/RIGID', noise_intensity=0.05,
                            batch_size=256, max_images=1000):
        """Run RIGID detection using the improved rigid_detector.py."""
        try:
            # Import the detect_images function from rigid_detector
            from rigid_detector import detect_images

            # Use defaults if not specified
            if real_datasets is None:
                real_datasets = self.default_real_datasets
            if ai_datasets is None:
                ai_datasets = self.default_ai_datasets

            logger.info("Starting RIGID detection...")
            logger.info(f"Real datasets: {real_datasets}")
            logger.info(f"AI datasets: {ai_datasets}")
            logger.info(f"Output directory: {output_dir}")

            # Run detection
            results = detect_images(
                dataset_paths=dataset_paths,
                real_datasets=real_datasets,
                ai_datasets=ai_datasets,
                output_dir=output_dir,
                noise_intensity=noise_intensity,
                batch_size=batch_size,
                max_images=max_images
            )

            return results

        except ImportError as e:
            logger.error(f"Could not import rigid_detector: {e}")
            logger.error("Make sure rigid_detector.py is in the RIGID directory")
            return None
        except Exception as e:
            logger.error(f"Error running RIGID detection: {e}")
            return None

    def save_individual_results(self, results, base_output_dir):
        """Save individual dataset results in SReC-compatible format."""
        saved_files = []

        for dataset_name, dataset_results in results.items():
            if dataset_name == 'metrics':
                continue

            # Create output directory for this dataset
            dataset_output_dir = Path(base_output_dir) / dataset_name.upper()
            os.makedirs(dataset_output_dir, exist_ok=True)

            # Create image list from actual filenames
            image_list_file = dataset_output_dir / "image_list.txt"
            image_similarities = dataset_results.get('image_similarities', {})

            with open(image_list_file, 'w') as f:
                for img_path in image_similarities.keys():
                    filename = os.path.basename(img_path)
                    f.write(f"{filename}\n")

            # Create results JSON in SReC format (already in correct format!)
            results_file = dataset_output_dir / f"{dataset_name}_rigid_results.json"

            # Use the image_similarities directly (SReC format: path -> similarity)
            with open(results_file, 'w') as f:
                json.dump(image_similarities, f, indent=2)

            saved_files.append(results_file)
            logger.info(f"Saved {dataset_name} results: {results_file}")

        return saved_files

    def validate_outputs(self, output_dir):
        """Validate that expected output files exist."""
        output_dir = Path(output_dir)

        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            return False

        # Look for JSON result files
        json_files = list(output_dir.rglob("*_rigid_results.json"))

        if not json_files:
            logger.error("No RIGID result files found")
            return False

        # Validate JSON files
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if not data:
                        logger.warning(f"Empty JSON file: {json_file}")
                    else:
                        logger.info(f"✓ Valid JSON file: {json_file} ({len(data)} entries)")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON file {json_file}: {e}")
                return False

        return True

    def run_pipeline(self, input_dirs, real_datasets=None, ai_datasets=None,
                     base_output_dir='/mnt/ssd-data/vaidya/combined_pipeline/results/RIGID', noise_intensity=0.05,
                     batch_size=256, max_images=1000):
        """Run the complete RIGID pipeline for multiple datasets."""
        start_time = time.time()
        logger.info("Starting RIGID Pipeline Execution")

        # Create dataset paths dictionary
        dataset_paths = {}
        for input_dir in input_dirs:
            input_path = Path(input_dir)
            if not input_path.exists():
                logger.error(f"Input directory does not exist: {input_dir}")
                continue
            dataset_name = input_path.name.lower()
            dataset_paths[dataset_name] = str(input_path)

        if not dataset_paths:
            logger.error("No valid input directories found")
            return False

        # Determine real vs AI datasets based on directory names if not specified
        if real_datasets is None:
            real_datasets = [name for name in dataset_paths.keys() if name in self.default_real_datasets]
        if ai_datasets is None:
            ai_datasets = [name for name in dataset_paths.keys() if name in self.default_ai_datasets]

        # Add any unclassified datasets to AI datasets with warning
        all_specified = set(real_datasets + ai_datasets)
        unclassified = set(dataset_paths.keys()) - all_specified
        if unclassified:
            logger.warning(f"Unclassified datasets (assuming AI): {list(unclassified)}")
            ai_datasets.extend(list(unclassified))

        # Run RIGID detection
        results = self.run_rigid_detection(
            dataset_paths=dataset_paths,
            real_datasets=real_datasets,
            ai_datasets=ai_datasets,
            output_dir=base_output_dir,
            noise_intensity=noise_intensity,
            batch_size=batch_size,
            max_images=max_images
        )

        if results is None:
            logger.error("RIGID detection failed")
            return False

        # Save individual dataset results
        saved_files = self.save_individual_results(results, base_output_dir)

        # Validate outputs
        success = self.validate_outputs(base_output_dir)

        # Print summary
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info("\n" + "=" * 60)
        logger.info("RIGID PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        logger.info(f"Processed datasets: {len(dataset_paths)}")
        logger.info(f"Output files: {len(saved_files)}")

        # Print per-dataset results
        for dataset_name, dataset_results in results.items():
            if dataset_name != 'metrics':
                logger.info(
                    f"{dataset_name:15s}: {dataset_results['mean_similarity']:.4f} ({dataset_results['dataset_type']}) - {dataset_results['num_images']} images")

        # Print overall metrics
        if 'metrics' in results:
            logger.info("\n" + "-" * 40)
            logger.info("OVERALL METRICS")
            logger.info("-" * 40)
            metrics = results['metrics']
            logger.info(f"AUROC:             {metrics['auroc']:.4f}")
            logger.info(f"FPR95:             {metrics['fpr95']:.4f}")
            logger.info(f"Average Precision: {metrics['average_precision']:.4f}")
            logger.info(f"Real images:       {metrics['num_real_images']}")
            logger.info(f"AI images:         {metrics['num_ai_images']}")

        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RIGID Pipeline Runner")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--images", type=str,
                             help="Single directory containing images to process")
    input_group.add_argument("--images-list", nargs='+',
                             help="List of directories containing images to process")

    # Dataset classification options
    parser.add_argument("--real-datasets", nargs='+',
                        help="List of dataset names that contain real images (default: coco, raise)")
    parser.add_argument("--ai-datasets", nargs='+',
                        help="List of dataset names that contain AI-generated images")

    # Output options
    parser.add_argument("--base-output", type=str, default="/mnt/ssd-data/vaidya/combined_pipeline/results/RIGID",
                        help="Base output directory for results (default: /mnt/ssd-data/vaidya/combined_pipeline/results/RIGID)")

    # RIGID parameters
    parser.add_argument("--noise-intensity", type=float, default=0.05,
                        help="Noise intensity parameter (lamb in notebook, default: 0.05)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for processing (default: 256)")
    parser.add_argument("--max-images", type=int, default=1000,
                        help="Maximum number of images to process per dataset (default: 1000)")

    args = parser.parse_args()

    # Determine input directories to process
    if args.images:
        input_dirs = [args.images]
    else:
        input_dirs = args.images_list

    # Initialize runner
    runner = RigidRunner()

    # Run pipeline
    success = runner.run_pipeline(
        input_dirs=input_dirs,
        real_datasets=args.real_datasets,
        ai_datasets=args.ai_datasets,
        base_output_dir=args.base_output,
        noise_intensity=args.noise_intensity,
        batch_size=args.batch_size,
        max_images=args.max_images
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())