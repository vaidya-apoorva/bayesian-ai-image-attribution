#!/usr/bin/env python3
"""
Resize all images in datasets to 512x512 for Aeroblade processing.
This ensures all images have uniform size to avoid tensor size mismatch errors.
"""

from PIL import Image
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def resize_dataset(input_dir, output_dir, size=(512, 512)):
    """Resize all images in input_dir to specified size and save to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg'))

    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return 0

    logger.info(f"Resizing {len(image_files)} images from {input_dir} to {size[0]}x{size[1]}")

    success_count = 0
    for img_path in tqdm(image_files, desc=f"Processing {input_path.name}"):
        try:
            # Open and resize image
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.LANCZOS)

            # Save to output directory with same filename
            output_file = output_path / img_path.name
            img_resized.save(output_file)
            success_count += 1

        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")

    logger.info(f"Successfully resized {success_count}/{len(image_files)} images")
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Resize all dataset images to 512x512")
    parser.add_argument("--size", type=int, default=512, help="Target size (creates square images)")
    parser.add_argument("--input-base", default="/mnt/hdd-data/vaidya/dataset",
                       help="Base directory containing datasets")
    parser.add_argument("--output-base", default="/mnt/hdd-data/vaidya/dataset_512",
                       help="Base directory for resized datasets")

    args = parser.parse_args()

    # Define all datasets to process
    datasets = [
        'coco',
        'dalle2',
        'dalle3',
        'firefly',
        'midjourneyV5',
        'midjourneyV6',
        'raise',
        'sdxl',
        'stable_diffusion_1_5'
    ]

    logger.info(f"Starting resize operation: {args.size}x{args.size}")
    logger.info(f"Input base: {args.input_base}")
    logger.info(f"Output base: {args.output_base}")
    logger.info(f"Datasets to process: {len(datasets)}")

    total_images = 0
    for dataset_name in datasets:
        input_dir = Path(args.input_base) / dataset_name
        output_dir = Path(args.output_base) / dataset_name

        if not input_dir.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            continue

        count = resize_dataset(input_dir, output_dir, size=(args.size, args.size))
        total_images += count

    logger.info("\n" + "="*60)
    logger.info("RESIZE SUMMARY")
    logger.info("="*60)
    logger.info(f"Total images resized: {total_images}")
    logger.info(f"Resized datasets saved to: {args.output_base}")
    logger.info("="*60)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
