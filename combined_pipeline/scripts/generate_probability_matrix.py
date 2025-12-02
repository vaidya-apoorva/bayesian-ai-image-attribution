#!/usr/bin/env python3
"""
Generate ResNet Probability Matrix

Test all trained binary ResNet classifiers on all datasets to create a
classifier×dataset probability matrix showing P(Real) and P(Generator)
for each combination.

Usage:
    python generate_probability_matrix.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
data_dir = '/mnt/hdd-data/vaidya/dataset'
models_dir = '/mnt/ssd-data/vaidya/combined_pipeline/models'
output_dir = '/mnt/ssd-data/vaidya/combined_pipeline/results'
os.makedirs(output_dir, exist_ok=True)

# Available datasets and generators (all 7 generators now trained)
available_generators = ['dalle2', 'dalle3', 'firefly', 'midjourneyV5', 'midjourneyV6', 'sdxl', 'stable_diffusion_1_5']
all_datasets = ['coco'] + available_generators  # Real + AI generators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_binary_classifier(generator_name):
    """Load a trained binary classifier (handles mixed ResNet18/ResNet50 architectures)."""

    # Try ResNet50 first (most generators)
    model_path = Path(models_dir) / f"resnet50_{generator_name}_vs_coco.pth"
    architecture = 'resnet50'

    # If not found, try ResNet18 (DALL-E 2)
    if not model_path.exists():
        model_path = Path(models_dir) / f"resnet18_{generator_name}_vs_coco.pth"
        architecture = 'resnet18'

    if not model_path.exists():
        print(f"Warning: Model not found for {generator_name}")
        return None

    # Create model architecture based on what we found
    if architecture == 'resnet50':
        model = models.resnet50(pretrained=False)
    else:  # resnet18
        model = models.resnet18(pretrained=False)

    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded {architecture} classifier: {generator_name} (Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.3f})")
    return model


def create_dataset(dataset_name, max_images=100):
    """Create dataset for testing."""
    dataset_path = Path(data_dir) / dataset_name

    if not dataset_path.exists():
        print(f"Warning: Dataset not found: {dataset_path}")
        return None

    # Get image files
    image_files = list(dataset_path.glob("*.png"))
    if len(image_files) == 0:
        print(f"Warning: No PNG images found in {dataset_path}")
        return None

    # Limit number of images for efficiency
    if len(image_files) > max_images:
        image_files = image_files[:max_images]

    print(f"Dataset {dataset_name}: {len(image_files)} images")
    return image_files


def test_classifier_on_dataset(model, image_files, dataset_name, classifier_name):
    """Test a binary classifier on a dataset."""
    if model is None or image_files is None:
        return None

    probabilities = []

    print(f"  Testing {classifier_name} on {dataset_name}...")

    with torch.no_grad():
        for img_path in image_files:
            try:
                # Load and transform image
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Get prediction
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)[0]  # [P(Generator), P(Real)] - alphabetical order

                probabilities.append({
                    'image': str(img_path.name),
                    'P_real': probs[1].item(),  # class 1 = "real"
                    'P_generator': probs[0].item(),  # class 0 = "generator"
                    'predicted_class': 'real' if probs[1] > probs[0] else 'generator'
                })

            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue

    return probabilities


def generate_probability_matrix():
    """Generate the complete probability matrix."""

    print("=" * 60)
    print("GENERATING RESNET PROBABILITY MATRIX")
    print("=" * 60)

    # Load all available binary classifiers
    classifiers = {}
    for generator in available_generators:
        classifier = load_binary_classifier(generator)
        if classifier is not None:
            classifiers[generator] = classifier

    print(f"\nLoaded {len(classifiers)} binary classifiers")

    # Create datasets
    datasets_dict = {}
    for dataset_name in all_datasets:
        dataset = create_dataset(dataset_name, max_images=100)
        if dataset is not None:
            datasets_dict[dataset_name] = dataset

    print(f"Created {len(datasets_dict)} datasets")

    # Generate probability matrix
    results = {}
    summary_matrix = {}

    for classifier_name, model in classifiers.items():
        print(f"\n{'=' * 40}")
        print(f"Testing classifier: {classifier_name}")
        print(f"{'=' * 40}")

        classifier_results = {}

        for dataset_name, image_files in datasets_dict.items():
            probabilities = test_classifier_on_dataset(
                model, image_files, dataset_name, classifier_name
            )

            if probabilities:
                classifier_results[dataset_name] = probabilities

                # Calculate summary statistics
                p_real_mean = np.mean([p['P_real'] for p in probabilities])
                p_gen_mean = np.mean([p['P_generator'] for p in probabilities])
                accuracy = np.mean([
                    1 if (p['predicted_class'] == 'real' and dataset_name == 'coco') or
                         (p['predicted_class'] == 'generator' and dataset_name != 'coco')
                    else 0 for p in probabilities
                ])

                if classifier_name not in summary_matrix:
                    summary_matrix[classifier_name] = {}

                summary_matrix[classifier_name][dataset_name] = {
                    'P_real_mean': p_real_mean,
                    'P_generator_mean': p_gen_mean,
                    'accuracy': accuracy,
                    'num_images': len(probabilities)
                }

                print(f"    {dataset_name:20s}: P(Real)={p_real_mean:.3f}, P(Gen)={p_gen_mean:.3f}, Acc={accuracy:.3f}")

        results[classifier_name] = classifier_results

    return results, summary_matrix


def save_results(results, summary_matrix):
    """Save results to JSON and CSV files."""

    # Save detailed results
    detailed_path = Path(output_dir) / "resnet_probability_matrix_detailed.json"
    with open(detailed_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {detailed_path}")

    # Save summary matrix
    summary_path = Path(output_dir) / "resnet_probability_matrix_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_matrix, f, indent=2)
    print(f"Summary matrix saved to: {summary_path}")

    # Create CSV matrices for easy analysis

    # P(Real) matrix
    p_real_matrix = []
    datasets = list(next(iter(summary_matrix.values())).keys())

    for classifier in summary_matrix.keys():
        row = [summary_matrix[classifier][dataset]['P_real_mean']
               for dataset in datasets]
        p_real_matrix.append(row)

    p_real_df = pd.DataFrame(p_real_matrix,
                             index=list(summary_matrix.keys()),
                             columns=datasets)

    p_real_csv = Path(output_dir) / "resnet_p_real_matrix.csv"
    p_real_df.to_csv(p_real_csv)
    print(f"P(Real) matrix saved to: {p_real_csv}")

    # P(Generator) matrix
    p_gen_matrix = []
    for classifier in summary_matrix.keys():
        row = [summary_matrix[classifier][dataset]['P_generator_mean']
               for dataset in datasets]
        p_gen_matrix.append(row)

    p_gen_df = pd.DataFrame(p_gen_matrix,
                            index=list(summary_matrix.keys()),
                            columns=datasets)

    p_gen_csv = Path(output_dir) / "resnet_p_generator_matrix.csv"
    p_gen_df.to_csv(p_gen_csv)
    print(f"P(Generator) matrix saved to: {p_gen_csv}")

    # Accuracy matrix
    acc_matrix = []
    for classifier in summary_matrix.keys():
        row = [summary_matrix[classifier][dataset]['accuracy']
               for dataset in datasets]
        acc_matrix.append(row)

    acc_df = pd.DataFrame(acc_matrix,
                          index=list(summary_matrix.keys()),
                          columns=datasets)

    acc_csv = Path(output_dir) / "resnet_accuracy_matrix.csv"
    acc_df.to_csv(acc_csv)
    print(f"Accuracy matrix saved to: {acc_csv}")

    return p_real_df, p_gen_df, acc_df


def create_visualizations(p_real_df, p_gen_df, acc_df):
    """Create heatmap visualizations."""

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # P(Real) heatmap
    sns.heatmap(p_real_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0.5, ax=axes[0], cbar_kws={'label': 'P(Real)'})
    axes[0].set_title('ResNet Binary Classifiers: P(Real)')
    axes[0].set_xlabel('Test Dataset')
    axes[0].set_ylabel('Classifier (Generator vs COCO)')

    # P(Generator) heatmap
    sns.heatmap(p_gen_df, annot=True, fmt='.3f', cmap='RdYlBu',
                center=0.5, ax=axes[1], cbar_kws={'label': 'P(Generator)'})
    axes[1].set_title('ResNet Binary Classifiers: P(Generator)')
    axes[1].set_xlabel('Test Dataset')
    axes[1].set_ylabel('Classifier (Generator vs COCO)')

    # Accuracy heatmap
    sns.heatmap(acc_df, annot=True, fmt='.3f', cmap='Greens',
                ax=axes[2], cbar_kws={'label': 'Accuracy'})
    axes[2].set_title('ResNet Binary Classifiers: Accuracy')
    axes[2].set_xlabel('Test Dataset')
    axes[2].set_ylabel('Classifier (Generator vs COCO)')

    plt.tight_layout()

    # Save visualization
    viz_path = Path(output_dir) / "resnet_probability_matrix_heatmaps.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {viz_path}")

    plt.show()


def main():
    """Main execution function."""

    print("Starting ResNet Probability Matrix Generation")
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Output directory: {output_dir}")

    # Generate probability matrix
    results, summary_matrix = generate_probability_matrix()

    if not results:
        print("No results generated. Check classifiers and datasets.")
        return

    # Save results
    p_real_df, p_gen_df, acc_df = save_results(results, summary_matrix)

    # Create visualizations
    create_visualizations(p_real_df, p_gen_df, acc_df)

    print(f"\n{'=' * 60}")
    print("PROBABILITY MATRIX GENERATION COMPLETED")
    print(f"{'=' * 60}")
    print(f"Classifiers tested: {len(summary_matrix)}")
    print(f"Datasets tested: {len(next(iter(summary_matrix.values())))}")
    print("Results ready for Bayesian integration!")


if __name__ == "__main__":
    main()