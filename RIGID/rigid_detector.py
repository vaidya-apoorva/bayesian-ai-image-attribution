#!/usr/bin/env python3
"""
RIGID Detector - Simple conversion from RIGID.ipynb

This is a direct conversion of the RIGID notebook to a Python script.
It contains the core RIGID detection functionality without any additional CLI features.
"""

import numpy as np
import random
import os
import sys
from sklearn import metrics
from pathlib import Path
from PIL import Image
import glob

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF

# Set random seeds for reproducibility
seed = 100
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class SRecStyleImageDataset(Dataset):
    """Dataset that mimics SReC's ImageFolder - takes directory + filename list."""

    def __init__(self, dir_path, filenames, transform=None):
        """
        Args:
            dir_path (str): Path to directory containing images
            filenames (list): List of image filenames in the directory
            transform: Optional transform to apply to images
        """
        self.dir_path = Path(dir_path)
        self.filenames = filenames
        self.transform = transform

        if not filenames:
            raise RuntimeError(f"Filename list is empty for {dir_path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = self.dir_path / filename

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return filename, image  # Return filename and image like SReC
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                return filename, self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            return filename, Image.new('RGB', (224, 224), (0, 0, 0))


def sim_auc(similarities, datasets):
    """
    Calculate AUC and FPR95 for multiple OOD datasets against ID dataset.

    Args:
        similarities (list): List of similarity arrays, first one is ID dataset
        datasets (list): List of dataset names

    Returns:
        tuple: (average_auc, average_fpr95)
    """
    if len(similarities) != len(datasets):
        raise ValueError("Number of similarities arrays must match number of dataset names")

    if len(similarities) < 2:
        raise ValueError("At least 2 datasets (ID and OOD) are required")

    similarities = np.array(similarities, dtype=object)  # Use object dtype for arrays of different lengths
    id_confi = similarities[0]

    auc_scores = []
    fpr_scores = []

    for ood_confi, dataset in zip(similarities[1:], datasets[1:]):
        auroc, fpr_95 = calculate_auc_metrics(id_confi, ood_confi)
        auc_scores.append(auroc)
        fpr_scores.append(fpr_95)
        print(f"Dataset: {dataset:<25} | AUC: {auroc:.4f} | FPR95: {fpr_95:.4f}")

    avg_auc = np.mean(auc_scores)
    avg_fpr = np.mean(fpr_scores)

    print("-" * 60)
    print(f"Average AUC: {avg_auc:.4f} | Average FPR95: {avg_fpr:.4f}")

    return avg_auc, avg_fpr


def sim_ap(similarities, datasets):
    """
    Calculate Average Precision for multiple OOD datasets against ID dataset.

    Args:
        similarities (list): List of similarity arrays, first one is ID dataset
        datasets (list): List of dataset names

    Returns:
        float: average AP score
    """
    if len(similarities) != len(datasets):
        raise ValueError("Number of similarities arrays must match number of dataset names")

    if len(similarities) < 2:
        raise ValueError("At least 2 datasets (ID and OOD) are required")

    similarities = np.array(similarities, dtype=object)
    id_confi = similarities[0]

    ap_scores = []

    for ood_confi, dataset in zip(similarities[1:], datasets[1:]):
        aver_p = calculate_average_precision(id_confi, ood_confi)
        ap_scores.append(aver_p)
        print(f"Dataset: {dataset:<25} | AP: {aver_p:.4f}")

    avg_ap = np.mean(ap_scores)
    print("-" * 40)
    print(f"Average AP: {avg_ap:.4f}")

    return avg_ap


def calculate_auc_metrics(id_conf, ood_conf):
    """
    Calculate AUROC and FPR at 95% TPR for binary classification.

    Args:
        id_conf (np.ndarray): Confidence scores for ID (in-distribution) samples
        ood_conf (np.ndarray): Confidence scores for OOD (out-of-distribution) samples

    Returns:
        tuple: (auroc, fpr_at_95_tpr)
    """
    # Combine predictions and create labels
    all_conf = np.concatenate([id_conf, ood_conf])
    # ID samples are positive (1), OOD samples are negative (0)
    labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])

    # Calculate ROC curve
    fpr, tpr, _ = metrics.roc_curve(labels, all_conf)

    # Calculate AUROC
    auroc = metrics.auc(fpr, tpr)

    # Calculate FPR at 95% TPR
    tpr_threshold = 0.95
    valid_indices = tpr >= tpr_threshold
    if np.any(valid_indices):
        fpr_at_95 = fpr[np.argmax(valid_indices)]
    else:
        fpr_at_95 = fpr[-1]
        print(f"Warning: 95% TPR not achievable. Max TPR: {tpr[-1]:.3f}")

    return auroc, fpr_at_95


def calculate_average_precision(id_predictions, ood_predictions):
    """
    Calculate Average Precision for binary classification.

    Args:
        id_predictions (np.ndarray): Predictions for ID samples
        ood_predictions (np.ndarray): Predictions for OOD samples

    Returns:
        float: Average Precision score
    """
    # Combine predictions and create labels
    all_predictions = np.concatenate([id_predictions, ood_predictions])
    # ID samples are positive (1), OOD samples are negative (0)
    labels = np.concatenate([np.ones(len(id_predictions)), np.zeros(len(ood_predictions))])

    # Calculate Average Precision
    average_precision = metrics.average_precision_score(labels, all_predictions)

    return average_precision


# Default normalization constants
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


class RIGID_Detector():
    """
    RIGID Detector class for AI-generated image detection.

    Uses DINOv2 ViT-L/14 model to compute similarity between original
    and noise-perturbed images.
    """

    def __init__(self, lamb=0.05, percentile=5):
        """
        Initialize RIGID detector.

        Args:
            lamb (float): Noise intensity parameter
            percentile (int): Percentile parameter (unused in current implementation)
        """
        self.lamb = lamb
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
        self.model.eval()

    @torch.no_grad()
    def calculate_sim(self, data):
        """
        Calculate similarity between original and noise-perturbed images.

        Args:
            data (torch.Tensor): Input images tensor

        Returns:
            torch.Tensor: Cosine similarity scores
        """
        features = self.model(data)
        noise = torch.randn_like(data).to(data.device)
        trans_data = data + noise * self.lamb
        trans_features = self.model(trans_data)
        sim_feat = F.cosine_similarity(features, trans_features, dim=-1)
        return sim_feat

    @torch.no_grad()
    def detect(self, data):
        """
        Detect AI-generated images by computing similarity scores.

        Args:
            data (torch.Tensor): Input images tensor

        Returns:
            torch.Tensor: Similarity scores
        """
        sim = self.calculate_sim(data)
        return sim


def get_image_filenames(dir_path):
    """Get list of image filenames in directory (like SReC does)."""
    dir_path = Path(dir_path)
    # Comprehensive list of image extensions (both lowercase and uppercase)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif',
                        '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF', '.WEBP', '.GIF'}

    filenames = []
    for ext in image_extensions:
        # Use rglob for recursive search (handles nested directories like ImageNet classes)
        found_files = [str(f.relative_to(dir_path)) for f in dir_path.rglob(f'*{ext}')]
        filenames.extend(found_files)
        if found_files:  # Debug info
            print(f"Found {len(found_files)} files with extension {ext}")

    if not filenames:
        # Additional debugging
        print(f"No images found in {dir_path}")
        print(f"Directory contents: {list(dir_path.iterdir())[:10]}...")  # Show first 10 items
        raise RuntimeError(f"No images found in {dir_path}")

    print(f"Total images found: {len(filenames)}")
    return sorted(filenames)  # Sort for consistent ordering


def detect_images(dataset_paths, real_datasets=None, ai_datasets=None,
                  output_dir='./results/RIGID', noise_intensity=0.05,
                  batch_size=256, max_images=1000):
    """
    Detect AI-generated images using RIGID method.

    Args:
        dataset_paths (dict): Dictionary mapping dataset names to their paths
        real_datasets (list): List of dataset names that contain real images
        ai_datasets (list): List of dataset names that contain AI-generated images
        output_dir (str): Directory to save results
        noise_intensity (float): Noise intensity parameter for RIGID
        batch_size (int): Batch size for processing
        max_images (int): Maximum number of images to process per dataset

    Returns:
        dict: Results containing similarity scores for each dataset
    """
    if real_datasets is None:
        real_datasets = ['coco']  # Only use COCO as real dataset
    if ai_datasets is None:
        ai_datasets = ['dalle2', 'dalle3', 'firefly', 'midjourneyV5', 'midjourneyV6', 'sdxl', 'stable_diffusion_1_5']

    # Transform for RIGID detection
    transform_RIGID = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])

    # Initialize RIGID detector
    rigid_detector = RIGID_Detector(lamb=noise_intensity)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Store results
    results = {}
    sim_datasets = []
    dataset_names = []
    dataset_types = []  # Track which datasets are real vs AI

    # Process all datasets
    all_datasets = list(dataset_paths.keys())

    print(f"Processing {len(all_datasets)} datasets...")
    print(f"Real datasets: {real_datasets}")
    print(f"AI datasets: {ai_datasets}")

    with torch.no_grad():
        for dataset_name in all_datasets:
            dataset_path = dataset_paths[dataset_name]

            # Determine dataset type
            if dataset_name in real_datasets:
                dataset_type = 'real'
            elif dataset_name in ai_datasets:
                dataset_type = 'ai'
            else:
                print(f"Warning: Dataset '{dataset_name}' not specified as real or AI. Assuming AI.")
                dataset_type = 'ai'

            dataset_types.append(dataset_type)
            dataset_names.append(dataset_name)

            print(f"Processing {dataset_name} ({dataset_type}) from {dataset_path}")

            try:
                # Get list of image filenames (like SReC does)
                filenames = get_image_filenames(dataset_path)
                print(f"Found {len(filenames)} images in {dataset_name}")

                # Use SReC-style dataset (directory + filename list)
                dataset_folder = SRecStyleImageDataset(dataset_path, filenames, transform=transform_RIGID)
                data_loader = DataLoader(dataset_folder, batch_size=batch_size, shuffle=True, num_workers=2)

                sim_feat = []
                image_paths = []
                total_num = 0
                for i, (filenames_batch, samples) in enumerate(data_loader):
                    samples = samples.cuda() if torch.cuda.is_available() else samples
                    samples_num = len(samples)
                    total_num += samples_num

                    # Store full image paths (like SReC format)
                    for filename in filenames_batch:
                        full_path = os.path.join(dataset_path, filename)
                        image_paths.append(full_path)

                    sim = rigid_detector.calculate_sim(samples)
                    sim_feat.append(sim)

                    if total_num >= max_images:
                        break

                if sim_feat:
                    sim_feat = torch.cat(sim_feat, dim=0)
                    similarities = sim_feat.cpu().numpy()
                    print(
                        f'{dataset_name} ({dataset_type}): {sim_feat.shape[0]} images, similarity: {sim_feat.mean().item():.4f}')

                    sim_datasets.append(similarities)

                    # Create image-similarity mapping like SReC format
                    image_similarities = {}
                    for img_path, sim_score in zip(image_paths[:len(similarities)], similarities):
                        image_similarities[img_path] = float(sim_score)

                    results[dataset_name] = {
                        'image_similarities': image_similarities,  # SReC-style path -> similarity mapping
                        'similarities': similarities.tolist(),
                        'mean_similarity': float(sim_feat.mean().item()),
                        'dataset_type': dataset_type,
                        'num_images': int(sim_feat.shape[0])
                    }
                else:
                    print(f"Warning: No images processed for {dataset_name}")

            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue

    # Calculate metrics if we have both real and AI datasets
    real_similarities = []
    ai_similarities = []

    for i, (dataset_name, dataset_type) in enumerate(zip(dataset_names, dataset_types)):
        if i < len(sim_datasets):
            if dataset_type == 'real':
                real_similarities.extend(sim_datasets[i])
            else:
                ai_similarities.extend(sim_datasets[i])

    if real_similarities and ai_similarities:
        real_similarities = np.array(real_similarities)
        ai_similarities = np.array(ai_similarities)

        print("\n" + "=" * 50)
        print("DETECTION METRICS")
        print("=" * 50)

        # Calculate AUC and FPR95
        auroc, fpr95 = calculate_auc_metrics(real_similarities, ai_similarities)
        print(f"AUROC: {auroc:.4f}")
        print(f"FPR95: {fpr95:.4f}")

        # Calculate Average Precision
        ap = calculate_average_precision(real_similarities, ai_similarities)
        print(f"Average Precision: {ap:.4f}")

        # Add metrics to results
        results['metrics'] = {
            'auroc': float(auroc),
            'fpr95': float(fpr95),
            'average_precision': float(ap),
            'num_real_images': len(real_similarities),
            'num_ai_images': len(ai_similarities)
        }
    else:
        print("Warning: Could not calculate metrics. Need both real and AI datasets.")

    return results


if __name__ == "__main__":
    # Example usage with default datasets - only used when running this file directly
    # When called from rigid_runner.py, the dataset_paths parameter is used instead
    dataset_paths = {
        'coco': './datasets/coco',
        'raise': './datasets/raise',
        'dalle2': './datasets/dalle2',
        'dalle3': './datasets/dalle3',
        'firefly': './datasets/firefly',
        'midjourneyV5': './datasets/midjourneyV5',
        'midjourneyV6': './datasets/midjourneyV6',
        'sdxl': './datasets/sdxl',
        'stable_diffusion_1_5': './datasets/stable_diffusion_1_5'
    }

    # Specify which datasets are real vs AI-generated
    real_datasets = ['coco', 'raise']
    ai_datasets = ['dalle2', 'dalle3', 'firefly', 'midjourneyV5', 'midjourneyV6', 'sdxl', 'stable_diffusion_1_5']

    # Run detection
    results = detect_images(
        dataset_paths=dataset_paths,
        real_datasets=real_datasets,
        ai_datasets=ai_datasets,
        noise_intensity=0.05,
        batch_size=256,
        max_images=1000
    )

    print("\nResults summary:")
    for dataset_name, dataset_results in results.items():
        if dataset_name != 'metrics':
            print(f"{dataset_name}: {dataset_results['mean_similarity']:.4f} ({dataset_results['dataset_type']})")