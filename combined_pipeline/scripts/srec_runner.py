#!/usr/bin/env python3
"""
SReC Pipeline Runner

Script to execute SReC srec_detector.py and validate outputs.
This is a pipeline runner for testing SReC functionality.

Usage:
    # Single dataset
    python srec_runner.py --images /path/to/images --model /path/to/model.pth

    # Multiple datasets
    python srec_runner.py --images-list /path/to/dataset1 /path/to/dataset2 --model /path/to/model.pth
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

# Add project paths to system path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SRecRunner:
    """Runner class for executing SReC ZED_updated.py script."""

    def __init__(self, model_type="openimages"):
        """Initialize the test.

        Args:
            model_type: Either 'imagenet' or 'openimages' to determine output paths
        """
        # Updated path to point to the correct SReC directory
        self.srec_dir = Path("/mnt/ssd-data/vaidya/SReC")
        self.model_type = model_type.upper()  # IMAGENET or OPENIMAGES

    def create_image_list(self, image_dir, output_file):
        """Create image list file for srec_detector.py."""
        logger.info(f"Creating image list from {image_dir}")

        image_dir = Path(image_dir)
        # Support both PNG and JPEG formats
        png_images = list(image_dir.glob("*.png"))
        jpg_images = list(image_dir.glob("*.jpg"))
        jpeg_images = list(image_dir.glob("*.jpeg"))
        images = png_images + jpg_images + jpeg_images

        if not images:
            logger.error(f"No images found in {image_dir}")
            return False

        with open(output_file, 'w') as f:
            for img in images:
                f.write(f"{img.name}\n")

        logger.info(f"Created image list with {len(images)} images: {output_file}")
        return True

    def run_zed(self, image_dir, image_list_file, model_path, output_dir,
                resblocks=3, n_feats=64, scale=3, K=10, crop=0):
        """Run srec_detector.py with the specified parameters."""
        logger.info("Running srec_detector.py...")

        # Prepare SReC command
        zed_script = self.srec_dir / "srec_detector.py"

        if not zed_script.exists():
            logger.error(f"ZED script not found: {zed_script}")
            return False

        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        # Convert model path to absolute path to avoid issues when changing working directory
        abs_model_path = str(Path(model_path).resolve())

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python3", str(zed_script),
            "--path", str(image_dir),
            "--file", str(image_list_file),
            "--resblocks", str(resblocks),
            "--n-feats", str(n_feats),
            "--scale", str(scale),
            "--load", abs_model_path,
            "--K", str(K),
            "--crop", str(crop),
            "--log-likelihood",
            "--save-path", str(output_dir)
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {self.srec_dir}")

        try:
            # Run with real-time output streaming for progress monitoring
            logger.info("Starting srec_detector.py execution with real-time progress...")

            process = subprocess.Popen(cmd, cwd=str(self.srec_dir),
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, universal_newlines=True)

            # Stream output in real-time
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
                    # Print progress in real-time
                    logger.info(f"ZED: {output.strip()}")

            # Wait for process to complete
            return_code = process.wait()

            if return_code == 0:
                logger.info("srec_detector.py completed successfully")
                return True
            else:
                logger.error(f"srec_detector.py failed with exit code {return_code}")
                return False

        except Exception as e:
            logger.error(f"Error running srec_detector.py: {e}")
            return False

    def run_srec_pipeline(self, image_dir, model_path, output_dir,
                          resblocks=3, n_feats=64, scale=3, K=10, crop=0):
        """Run the complete SReC pipeline and validate outputs."""
        start_time = time.time()
        logger.info("Starting SReC Pipeline Execution")

        # Convert to Path objects
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)

        # Get dataset name from input directory
        dataset_name = image_dir.name

        # Create output directory first
        os.makedirs(output_dir, exist_ok=True)

        # Create image list file
        image_list_file = output_dir / "image_list.txt"
        success = self.create_image_list(image_dir, image_list_file)
        if not success:
            logger.error("Failed to create image list")
            return False

        # Run ZED
        success = self.run_zed(image_dir, image_list_file, model_path, output_dir,
                               resblocks, n_feats, scale, K, crop)
        if not success:
            logger.error("ZED execution failed")
            return False

        # Check for output files and validate
        success = self.validate_outputs(output_dir)
        if not success:
            logger.error("Output validation failed")
            return False

        # Post-process outputs: organize files into specified directories
        success = self.organize_outputs(output_dir, dataset_name)
        if not success:
            logger.error("Output organization failed")
            return False

        total_time = time.time() - start_time
        logger.info(f"SReC pipeline completed successfully in {total_time:.2f} seconds")
        return True

    def validate_outputs(self, output_dir):
        """Validate that expected output files were created with reasonable content."""
        logger.info("Validating output files...")
        output_dir = Path(output_dir)

        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            return False

        files = list(output_dir.iterdir())
        logger.info(f"Output directory contents ({len(files)} files):")

        # Check for expected output files
        expected_files = ["image_list.txt"]  # Files we expect to be created
        json_files = []
        found_files = []

        for file in sorted(files):
            size = file.stat().st_size if file.is_file() else "DIR"
            logger.info(f"  {file.name} ({size} bytes)")
            found_files.append(file.name)

            if file.suffix == ".json":
                json_files.append(file)

        # Validate expected files are present
        missing_files = [f for f in expected_files if f not in found_files]
        if missing_files:
            logger.warning(f"Missing expected files: {missing_files}")

        # Validate we have at least a JSON results file
        if not json_files:
            logger.error("No JSON output files found - srec_detector.py should produce results JSON")
            return False

        # Validate JSON files are valid
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                logger.info(f"✓ {json_file.name} is valid JSON with {len(data)} entries")
            except Exception as e:
                logger.error(f"✗ {json_file.name} is invalid JSON: {e}")
                return False

        return True

    def organize_outputs(self, output_dir, dataset_name):
        """Organize outputs into specified directory structure."""
        logger.info(f"Organizing outputs for dataset: {dataset_name}")

        output_dir = Path(output_dir)

        # Define target directories based on model type
        model_dir = f"{self.model_type}_MODEL"
        json_target_dir = Path(f"/mnt/ssd-data/vaidya/combined_pipeline/results/OTHER_DATASET/SREC/{model_dir}")
        srec_images_target_dir = Path(f"/mnt/ssd-data/vaidya/combined_pipeline/results/SREC/{model_dir}")

        # Create target directories
        os.makedirs(json_target_dir, exist_ok=True)
        os.makedirs(srec_images_target_dir, exist_ok=True)

        try:
            # Copy *_d0.json files to JSON target directory
            d0_json_files = list(output_dir.glob("*_d0.json"))
            for json_file in d0_json_files:
                target_json_path = json_target_dir / json_file.name
                shutil.copy2(json_file, target_json_path)
                logger.info(f"Copied {json_file.name} to {json_target_dir}")

            # Copy .srec image files to SReC images target directory
            srec_files = list(output_dir.glob("*.srec"))
            for srec_file in srec_files:
                target_srec_path = srec_images_target_dir / srec_file.name
                shutil.copy2(srec_file, target_srec_path)
                logger.info(f"Copied {srec_file.name} to {srec_images_target_dir}")

            logger.info(f"Successfully organized outputs for {dataset_name}")
            logger.info(f"  JSON files ({len(d0_json_files)}): {json_target_dir}")
            logger.info(f"  SReC files ({len(srec_files)}): {srec_images_target_dir}")

            return True

        except Exception as e:
            logger.error(f"Error organizing outputs: {e}")
            return False


def main():
    """Main entry point for the SReC pipeline runner."""
    parser = argparse.ArgumentParser(description="SReC Pipeline Runner")

    # Allow multiple input paths
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Single directory containing images to analyze")
    group.add_argument("--images-list", nargs='+', help="Multiple directories containing images to analyze")

    parser.add_argument("--model", required=True, help="Path to SReC model file (.pth)")
    parser.add_argument("--model-type", choices=['imagenet', 'openimages'], default='openimages',
                        help="Model type for organizing outputs (default: openimages)")
    parser.add_argument("--base-output", default="/mnt/ssd-data/vaidya/combined_pipeline/results/SREC",
                        help="Base output directory (default: /mnt/ssd-data/vaidya/combined_pipeline/results/SREC)")

    # Model architecture parameters
    parser.add_argument("--resblocks", type=int, default=5, help="Number of residual blocks (default: 5)")
    parser.add_argument("--n-feats", type=int, default=64, help="Number of features (default: 64)")
    parser.add_argument("--scale", type=int, default=3, help="Scale factor (default: 3)")
    parser.add_argument("--K", type=int, default=10, help="K parameter (default: 10)")
    parser.add_argument("--crop", type=int, default=0, help="Crop parameter (default: 0)")

    args = parser.parse_args()

    # Determine input directories to process
    if args.images:
        input_dirs = [args.images]
    else:
        input_dirs = args.images_list

    # Validate model file exists
    if not Path(args.model).exists():
        logger.error(f"Model file does not exist: {args.model}")
        return 1

    # Initialize runner with model type
    runner = SRecRunner(model_type=args.model_type)

    # Process each input directory
    all_success = True
    results_summary = {}

    for input_dir in input_dirs:
        # Validate input directory
        if not Path(input_dir).exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            all_success = False
            continue

        # Create dynamic output path based on dataset name
        dataset_name = Path(input_dir).name  # Get directory name (e.g., 'sdxl', 'dalle3', 'coco')
        output_dir = Path(args.base_output) / dataset_name.upper()

        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")

        # Run pipeline for this dataset
        success = runner.run_srec_pipeline(
            input_dir, args.model, output_dir,
            args.resblocks, args.n_feats, args.scale, args.K, args.crop
        )

        if success:
            results_summary[dataset_name] = str(output_dir)
            logger.info(f"✓ Successfully processed {dataset_name}")
        else:
            all_success = False
            logger.error(f"✗ Failed to process {dataset_name}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    for dataset, output_path in results_summary.items():
        logger.info(f"✓ {dataset}: {output_path}")

    if not all_success:
        logger.warning("Some datasets failed to process. Check logs above.")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())