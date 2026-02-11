#!/usr/bin/env python3
"""
Compare SREC, RIGID, and Aeroblade priors for Bayesian attribution.
Runs Bayesian attribution using each prior method and generates comparison reports.
"""

import subprocess
import sys
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_BASE = "/mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder"
RESULTS_BASE = "/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results_with_new_srec_weights/bayesian_pipeline_results"
SCRIPT_DIR = "/mnt/ssd-data/vaidya/combined_pipeline/scripts"
ANALYSIS_SCRIPT = f"{SCRIPT_DIR}/BAYESIAN_SCRIPTS/analyze_bayesian_performance.py"
BAYESIAN_ATTRIBUTION_SCRIPT = f"{SCRIPT_DIR}/BAYESIAN_SCRIPTS/bayesian_attribution_srec_weights.py"
# Limit number of images processed per dataset (set to None to process all)
MAX_IMAGES = None  # Process all 800 images instead of limiting to 400


def run_bayesian_attribution(method, model=None):
    """Run Bayesian attribution with specified prior method."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running {method.upper()}-based Bayesian attribution")
    logger.info(f"{'=' * 60}")

    # Process each dataset directory separately and combine results
    datasets = ['coco', 'dall-e2', 'dall-e3', 'firefly', 'midjourneyV5',
                'midjourneyV6', 'sdxl', 'stable_diffusion_1-5']

    all_results = []

    for dataset in datasets:
        dataset_dir = f"{INPUT_BASE}/{dataset}"
        if not Path(dataset_dir).exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue

        logger.info(f"Processing {dataset}...")

        cmd = [
            "python", BAYESIAN_ATTRIBUTION_SCRIPT,
            "--batch",
            "--input-dir", dataset_dir,
            "--method", method,
            "--output", f"{RESULTS_BASE}/temp_{method}_{dataset}_results.json"
        ]
        # Pass max images limit to the attributor script when configured
        if MAX_IMAGES is not None:
            cmd += ["--max-images", str(MAX_IMAGES)]

        if model and method == 'srec':
            cmd.extend(["--model", model])

        logger.info(f"Bayesian attribution command: {cmd}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to process {dataset}: {e}")
            return False

    # Combine all results
    combined_results = []
    for dataset in datasets:
        temp_file = Path(f"{RESULTS_BASE}/temp_{method}_{dataset}_results.json")
        if temp_file.exists():
            with open(temp_file, 'r') as f:
                data = json.load(f)
                combined_results.extend(data)
            temp_file.unlink()  # Delete temp file

    # Save combined results
    output_file = f"{RESULTS_BASE}/bayesian_{method}_results.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    logger.info(f"✓ {method.upper()} completed successfully ({len(combined_results)} images)")
    return True


def analyze_results(method):
    """Analyze results for a specific method."""
    logger.info(f"\nAnalyzing {method.upper()} results...")

    results_file = f"{RESULTS_BASE}/bayesian_{method}_results.json"
    output_dir = f"{RESULTS_BASE}/bayesian_performance_analysis/{method}"

    if not Path(results_file).exists():
        logger.warning(f"Results file not found: {results_file}")
        return False

    cmd = [
        "python", ANALYSIS_SCRIPT,
        "--results", results_file,
        "--output", output_dir
    ]
    logger.info(f"Analyse results command: {cmd}")
    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"✓ Analysis complete: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Analysis failed with exit code {e.returncode}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("BAYESIAN ATTRIBUTION COMPARISON")
    logger.info("=" * 60)
    logger.info(f"Input: {INPUT_BASE}")
    logger.info(f"Output: {RESULTS_BASE}")
    logger.info("=" * 60)

    # Ensure results directory exists
    Path(RESULTS_BASE).mkdir(parents=True, exist_ok=True)

    results = {}

    # Method 1: SREC (OpenImages model)
    logger.info("\n[1/3] SREC-based priors...")
    results['srec'] = run_bayesian_attribution('srec', model='openimages')

    # Method 2: RIGID
    logger.info("\n[2/3] RIGID-based priors...")
    results['rigid'] = run_bayesian_attribution('rigid')

    # Method 3: Aeroblade
    logger.info("\n[3/3] Aeroblade-based priors...")
    results['aeroblade'] = run_bayesian_attribution('aeroblade')

    # Analyze all results
    logger.info("\n" + "=" * 60)
    logger.info("ANALYZING RESULTS")
    logger.info("=" * 60)

    for method, success in results.items():
        if success:
            analyze_results(method)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for method, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{method.upper():15s}: {status}")

    logger.info("\n" + "=" * 60)
    logger.info("Results saved to:")
    logger.info(f"  {RESULTS_BASE}/")
    logger.info("\nAnalysis reports:")
    for method in results.keys():
        logger.info(f"  {RESULTS_BASE}/bayesian_performance_analysis/{method}/")
    logger.info("=" * 60)

    # Exit with error if any method failed
    if not all(results.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
