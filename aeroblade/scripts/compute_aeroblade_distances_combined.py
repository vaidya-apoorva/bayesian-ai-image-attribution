"""Compute the AE reconstruction distance."""

import argparse
from pathlib import Path

from aeroblade.high_level_funcs import compute_distances
from aeroblade.misc import safe_mkdir


def main(args):
    # If the path is a directory, collect all immediate subdirectories inside it
    if args.image_dir.is_dir():
        subdirs = [p for p in args.image_dir.iterdir() if p.is_dir()]
    else:
        subdirs = [args.image_dir]

    print(f"Processing image directories: {subdirs}")
    print(f"Using autoencoders: {args.autoencoders}")
    print(f"Distance metric: {args.distance_metric}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")

    # create output directory
    safe_mkdir(args.output_dir)

    # compute distances across all subdirectories (each with images)
    distances = compute_distances(
        dirs=subdirs,
        transforms=["clean"],
        repo_ids=args.autoencoders,
        distance_metrics=[args.distance_metric],
        amount=None,
        reconstruction_root=args.output_dir / "reconstructions",
        seed=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # save and display results
    csv_file_path = args.output_dir / f"{args.distance_metric}_distances.csv"
    distances.to_csv(csv_file_path, index=False)
    print(distances)
    print(f"\nSaved distances to {csv_file_path}\nDone.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the AE reconstruction distances for images or directories."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Path to an image directory or a directory containing multiple image subdirectories.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="aeroblade_output", help="Output directory."
    )
    parser.add_argument(
        "--autoencoders",
        nargs="+",
        default=[
            "CompVis/stable-diffusion-v1-1",  # SD1
            "stabilityai/stable-diffusion-2-base",  # SD2
            "kandinsky-community/kandinsky-2-1",  # KD2.1
        ],
        help="HuggingFace model names of autoencoder generators to use for reconstruction.",
    )
    parser.add_argument(
        "--distance-metric",
        default="lpips_vgg_2",
        choices=[
            "lpips_vgg_0",
            "lpips_vgg_1",
            "lpips_vgg_2",
            "lpips_vgg_3",
            "lpips_vgg_4",
            "lpips_vgg_5",
            "lpips_vgg_-1",
        ],
        help="Distance metric to use.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
