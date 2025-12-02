"""
ZED (Zero-shot Evidence Detector) - Updated Version

This script implements the SReC (Super-Resolution Compression) method for detecting
AI-generated images through lossless compression analysis. It computes D(l) values
(NLL - Entropy) which serve as evidence for AI generation.

Key concept: AI-generated images have different compression characteristics compared
to real images, which can be detected through super-resolution compression analysis.
"""

import os
import json
import torch
import click
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src import configs, data as lc_data, network
from src.l3c import bitcoding, timer

@click.command()
@click.option("--path", type=click.Path(exists=True), help="Directory of images.")
@click.option("--file", type=click.File("r"), help="File for image names.")
@click.option("--resblocks", type=int, default=5, show_default=True, help="Number of resblocks to use.")
@click.option("--n-feats", type=int, default=64, show_default=True, help="Size of n_feats vector used.")
@click.option("--scale", type=int, default=3, show_default=True, help="Scale of downsampling")
@click.option("--load", type=click.Path(exists=True), help="Path to load model")
@click.option("--K", type=int, default=10, help="Number of clusters in logistic mixture model.")
@click.option("--crop", type=int, default=0, help="Size of image crops in training. 0 means no crop.")
@click.option("--log-likelihood", is_flag=True, default=False, help="Turn on log-likelihood calculations.")
@click.option("--save-path", type=str, help="Save directory for images.")
def main(path, file, resblocks, n_feats, scale, load, k, crop, log_likelihood, save_path):
    """
    Main function for Zero-shot Evidence Detection using Super-Resolution Compression.

    This function:
    1. Loads a pre-trained super-resolution compression model
    2. Processes images through the compression pipeline
    3. Computes D(l) = NLL - Entropy for each image
    4. Saves results as JSON and creates visualization plots

    Args:
        path: Directory containing input images
        file: Text file with list of image filenames
        resblocks: Number of residual blocks in the network
        n_feats: Feature vector size
        scale: Downsampling scale factor
        load: Path to pre-trained model checkpoint
        k: Number of clusters in logistic mixture model
        crop: Crop size (0 = no cropping)
        log_likelihood: Whether to calculate log-likelihood
        save_path: Directory to save compressed files
    """

    # Configure model parameters
    # These settings define the super-resolution network architecture
    configs.n_feats = n_feats
    configs.resblocks = 5  # Fixed to 5 regardless of input (standard architecture)
    configs.K = k  # Number of mixture components for entropy modeling
    configs.scale = scale  # Super-resolution scale factor
    configs.log_likelihood = log_likelihood  # Enable likelihood computation
    configs.collect_probs = True  # Collect probability distributions for analysis

    # Load pre-trained super-resolution compression model
    # This model was trained to compress images through super-resolution
    checkpoint = torch.load(load)
    print(f"Loaded model from {load}.")
    print("Epoch:", checkpoint["epoch"])

    # Initialize the compressor network
    # The compressor combines super-resolution and compression
    compressor = network.Compressor()
    compressor.nets.load_state_dict(checkpoint["nets"])

    # Try to use GPU, fallback to CPU if CUDA fails
    try:
        compressor = compressor.cuda()  # Move to GPU for faster processing
        device = "cuda"
        print("Using GPU (CUDA)")
    except Exception as e:
        print(f"CUDA failed ({e}), falling back to CPU")
        compressor = compressor.cpu()
        device = "cpu"

    compressor.eval()  # Set to evaluation mode (disable dropout, batch norm updates)

    # Define image preprocessing transformations
    # These prepare images for the compression pipeline
    transforms = []
    if crop > 0:
        # Center crop to specified size if requested
        transforms.insert(0, T.CenterCrop(crop))
    transform = T.Compose(transforms)

    # Load dataset of images to process
    # ImageFolder creates a dataset from directory + filename list
    dataset = lc_data.ImageFolder(path, [filename.strip() for filename in file], scale, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Loaded directory with {len(dataset)} images.")

    # Create output directory for compressed files
    os.makedirs(save_path, exist_ok=True)

    # Create subdirectory for .srec images to keep output organized
    srec_images_dir = os.path.join(save_path, "srec-images")
    os.makedirs(srec_images_dir, exist_ok=True)

    # Initialize the bit-coding system for compression
    # This handles the actual compression encoding/decoding
    coder = bitcoding.Bitcoding(compressor)
    encoder_time_accumulator = timer.TimeAccumulator()

    # Dictionary to store D(l) results for each image
    # Key: full image path, Value: D(l) = NLL - Entropy
    results = {}

    # Process each image individually
    print("Processing images through compression pipeline...")
    for i, (filenames, x) in enumerate(dataloader):
        assert len(filenames) == 1, filenames
        filename = filenames[0]

        # Create safe file ID by removing extension and flattening path
        # Handle nested directories by using only the filename part
        filename_only = os.path.basename(filename)
        file_id = filename_only.rsplit(".", 1)[0]  # safer split than basic split()
        filepath = os.path.join(srec_images_dir, f"{file_id}.srec")

        # Clean up any existing compressed file
        if os.path.isfile(filepath):
            os.remove(filepath)

        # Encode the image through super-resolution compression
        # This is where the core SReC algorithm runs
        with encoder_time_accumulator.execute():
            # Move tensor to the same device as the model
            x = x.to(device)
            bits, entropy_coding_bytes, nll_list, entropy_list = coder.encode(x, filepath)

        # Calculate D(l) for the full resolution (level 0)
        # D(l) = NLL - Entropy is the key metric for AI detection
        # Higher D(l) values typically indicate AI-generated content
        print(f"Calculating D(l) for {filename} at level 0...")
        print("NLL:", nll_list[-1])
        print("Entropy:", entropy_list[-1])
        D_l = nll_list[-1] - entropy_list[-1]

        # Store result with full image path as key
        full_img_path = os.path.join(path, filename)
        results[full_img_path] = D_l
        print(f"D(l) = {D_l:.4f}")

    print("Encoding completed.")
    print("Results:", results)

    # Save D(l) results to JSON file for downstream processing
    # This JSON will be used by the Bayesian integration pipeline
    # Use the save_path directory and create dynamic filename based on input
    dataset_name = os.path.basename(path)  # Extract dataset name from path
    json_out = os.path.join(save_path, f"{dataset_name}_d0.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_out}")

    print("Visualization skipped - processing complete.")

if __name__ == "__main__":
    """
    Entry point for the ZED script.
    
    Example usage:
    python ZED_updated.py --path /path/to/images --file image_list.txt 
                         --load model.pth --save-path /path/to/output
                         
    python3 ZED_updated.py --path /mnt/hdd-data/vaidya/dataset/dalle3 
    --file /mnt/ssd-data/vaidya/SReC/datasets/dalle3_test.txt 
    --resblocks 3 --n-feats 64 --scale 3 
    --load /mnt/ssd-data/vaidya/SReC/models/openimages.pth 
    --K 10 --crop 0 --log-likelihood 
    --save-path /mnt/ssd-data/vaidya/SReC/srec-images/
    
    This will:
    1. Load images from the specified directory
    2. Process them through the super-resolution compression pipeline
    3. Compute D(l) values for AI detection
    4. Save results as JSON and create generator wise visualisation plots
    """
    main()
