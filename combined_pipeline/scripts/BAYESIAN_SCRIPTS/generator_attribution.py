#!/usr/bin/env python3
"""
Bayesian Generator Attribution

Compute Bayesian posteriors P(Generator|Image) using:
- Likelihoods from trained ResNet binary classifiers
- Priors from SReC D(l) values (or uniform if not available)

Usage:
    python bayesian_attribution.py --image path/to/image.png
    python bayesian_attribution.py --batch --input-dir path/to/images/
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
import argparse
import logging
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
models_dir = '/mnt/ssd-data/vaidya/combined_pipeline/models'
results_dir = '/mnt/ssd-data/vaidya/combined_pipeline/results/BAYESIAN_RESULTS'
srec_results_dir = '/mnt/ssd-data/vaidya/combined_pipeline/results/OTHER_DATASET/SREC/OPENIMAGES_MODEL'  # SReC D(l) JSON files

# Available generators
available_generators = ['dalle2', 'dalle3', 'firefly', 'midjourneyV5', 'midjourneyV6', 'sdxl', 'stable_diffusion_1_5']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class BayesianAttributor:
    """Bayesian generator attribution using binary classifiers and SReC priors."""

    def __init__(self, use_srec_priors=True):
        self.use_srec_priors = use_srec_priors
        self.classifiers = self.load_binary_classifiers()
        self.priors = self.load_priors()

        logger.info(f"Loaded {len(self.classifiers)} binary classifiers")
        logger.info(f"Using {'SReC' if use_srec_priors else 'uniform'} priors")

    def load_binary_classifiers(self):
        """Load all trained binary classifiers."""
        classifiers = {}

        for generator in available_generators:
            # Try ResNet50 first, then ResNet18
            model_path = Path(models_dir) / f"resnet50_{generator}_vs_coco.pth"
            architecture = 'resnet50'

            if not model_path.exists():
                model_path = Path(models_dir) / f"resnet18_{generator}_vs_coco.pth"
                architecture = 'resnet18'

            if not model_path.exists():
                logger.warning(f"Model not found for {generator}")
                continue

            # Create and load model
            if architecture == 'resnet50':
                model = models.resnet50(pretrained=False)
            else:
                model = models.resnet18(pretrained=False)

            model.fc = nn.Linear(model.fc.in_features, 2)

            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            classifiers[generator] = model
            logger.info(f"Loaded {architecture} classifier: {generator}")

        return classifiers

    def load_priors(self):
        """Load or create priors P(Generator)."""
        if self.use_srec_priors:
            # Try to load SReC results and convert to priors
            try:
                priors = self.load_srec_priors()
                logger.info("Loaded SReC-based priors")
                return priors
            except Exception as e:
                logger.warning(f"Could not load SReC priors: {e}")
                logger.info("Falling back to uniform priors")

        # Uniform priors as fallback
        uniform_prior = 1.0 / len(available_generators)
        priors = {gen: uniform_prior for gen in available_generators}
        priors['coco'] = uniform_prior  # Add real class

        logger.info("Using uniform priors")
        return priors

    def load_srec_priors(self):
        """Load SReC D(l) values and convert to priors."""
        # Map JSON filenames to our generator names
        file_mapping = {
            'coco_d0.json': 'coco',
            'dalle2_d0.json': 'dalle2',
            'dalle3_d0.json': 'dalle3',
            'firefly_d0.json': 'firefly',
            'midjourneyV5_d0.json': 'midjourneyV5',
            'midjourneyV6_d0.json': 'midjourneyV6',
            'sdxl_d0.json': 'sdxl',
            'stable_diffusion_1_5_d0.json': 'stable_diffusion_1_5'
        }

        # Load D(l) values for each dataset
        dl_values = {}
        logger.info("Loading SReC D(l) values from individual JSON files:")

        for json_filename, generator in file_mapping.items():
            json_path = Path(srec_results_dir) / json_filename

            if not json_path.exists():
                logger.warning(f"SReC file not found: {json_path}")
                dl_values[generator] = 0.0  # Default neutral value
                continue

            try:
                with open(json_path, 'r') as f:
                    image_dl_data = json.load(f)

                # Compute mean D(l) across all images in this dataset
                dl_list = list(image_dl_data.values())
                mean_dl = np.mean(dl_list)
                std_dl = np.std(dl_list)

                dl_values[generator] = mean_dl
                logger.info(f"  {generator:20s}: D(l) = {mean_dl:7.4f} ± {std_dl:6.4f} ({len(dl_list):4d} images)")

            except Exception as e:
                logger.error(f"Error loading {json_filename}: {e}")
                dl_values[generator] = 0.0

        # Add missing generators with default values
        for generator in available_generators + ['coco']:
            if generator not in dl_values:
                dl_values[generator] = 0.0
                logger.warning(f"Using default D(l)=0.0 for {generator}")

        # Convert D(l) values to priors
        # Higher D(l) = harder to compress = more likely to be AI-generated
        # Lower D(l) = easier to compress = more likely to be real

        # Separate real and AI generators
        coco_dl = dl_values['coco']
        ai_generators = [gen for gen in available_generators]
        ai_dl_values = [dl_values[gen] for gen in ai_generators]

        logger.info(f"\nD(l) statistics:")
        logger.info(f"  Real (COCO):     {coco_dl:7.4f}")
        logger.info(f"  AI generators:   {np.mean(ai_dl_values):7.4f} ± {np.std(ai_dl_values):6.4f}")

        # Use softmax to convert D(l) to priors
        # For AI generators: higher D(l) = higher prior probability
        alpha = 1.0  # Temperature parameter

        # Compute priors for AI generators
        ai_dl_array = np.array(ai_dl_values)
        ai_exp_vals = np.exp(ai_dl_array * alpha)
        ai_exp_sum = np.sum(ai_exp_vals)

        priors = {}
        for i, generator in enumerate(ai_generators):
            priors[generator] = ai_exp_vals[i] / ai_exp_sum

        # For real images: use inverse relationship with D(l)
        # Lower D(l) = higher probability of being real
        real_strength = np.exp(-alpha * coco_dl)
        ai_total_strength = np.sum(ai_exp_vals)

        # Balance real vs AI priors
        total_strength = real_strength + ai_total_strength
        real_prior = real_strength / total_strength
        ai_scale = (1.0 - real_prior)

        priors['coco'] = real_prior
        for gen in ai_generators:
            priors[gen] *= ai_scale

        logger.info("\nConverted to priors:")
        for gen, prior in sorted(priors.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {gen:20s}: P = {prior:6.3f}")

        return priors

    def get_likelihoods(self, image_path):
        """Get P(Image|Generator) from all binary classifiers."""
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        likelihoods = {}

        with torch.no_grad():
            for generator, model in self.classifiers.items():
                try:
                    outputs = model(image_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]

                    # P(Generator) from binary classifier
                    likelihood = probs[0].item()  # class 0 = generator (alphabetical order)
                    likelihoods[generator] = likelihood

                except Exception as e:
                    logger.warning(f"Error with {generator} classifier: {e}")
                    likelihoods[generator] = 0.1  # Default low likelihood

        # Add P(Image|Real) as geometric mean of P(Real) from all classifiers
        real_likelihoods = []
        with torch.no_grad():
            for generator, model in self.classifiers.items():
                try:
                    outputs = model(image_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    real_prob = probs[1].item()  # class 1 = real
                    real_likelihoods.append(real_prob)
                except:
                    real_likelihoods.append(0.5)

        # Geometric mean for P(Image|Real)
        if real_likelihoods:
            real_likelihood = np.prod(real_likelihoods) ** (1.0 / len(real_likelihoods))
        else:
            real_likelihood = 0.1

        likelihoods['coco'] = real_likelihood

        return likelihoods

    def compute_posteriors(self, image_path):
        """Compute Bayesian posteriors P(Generator|Image)."""
        # Get likelihoods P(Image|Generator)
        likelihoods = self.get_likelihoods(image_path)

        # Compute unnormalized posteriors: P(Generator|Image) ∝ P(Image|Generator) × P(Generator)
        log_posteriors = {}
        eps = 1e-12  # Numerical stability

        for generator in available_generators + ['coco']:
            likelihood = likelihoods.get(generator, eps)
            prior = self.priors.get(generator, eps)

            # Compute in log space for numerical stability
            log_posterior = np.log(likelihood + eps) + np.log(prior + eps)
            log_posteriors[generator] = log_posterior

        # Convert back to probabilities and normalize
        max_log = max(log_posteriors.values())
        exp_posteriors = {k: np.exp(v - max_log) for k, v in log_posteriors.items()}

        total = sum(exp_posteriors.values())
        posteriors = {k: v / total for k, v in exp_posteriors.items()}

        return {
            'image_path': str(image_path),
            'likelihoods': likelihoods,
            'priors': self.priors,
            'posteriors': posteriors,
            'predicted_generator': max(posteriors.items(), key=lambda x: x[1])[0],
            'confidence': max(posteriors.values())
        }

    def attribute_single_image(self, image_path):
        """Attribution for a single image."""
        logger.info(f"Attributing: {image_path}")
        return self.compute_posteriors(image_path)

    def attribute_batch(self, input_dir, output_file=None):
        """Attribution for all images in a directory."""
        input_path = Path(input_dir)
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))

        if not image_files:
            logger.error(f"No images found in {input_dir}")
            return None

        logger.info(f"Processing {len(image_files)} images...")

        results = []
        for img_file in image_files:
            try:
                result = self.compute_posteriors(img_file)
                results.append(result)

                if len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)}/{len(image_files)} images")

            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                continue

        # Save results
        if output_file is None:
            output_file = Path(results_dir) / "bayesian_attribution_20_images_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved {len(results)} attribution results to {output_file}")

        # Generate summary statistics
        self.generate_summary(results, output_file.parent / "20_imgaes_attribution_summary.json")

        return results

    def generate_summary(self, results, summary_file):
        """Generate summary statistics from attribution results."""
        # Count predictions per generator
        predictions = {}
        confidences = {}

        for result in results:
            pred = result['predicted_generator']
            conf = result['confidence']

            if pred not in predictions:
                predictions[pred] = 0
                confidences[pred] = []

            predictions[pred] += 1
            confidences[pred].append(conf)

        # Calculate summary stats
        summary = {
            'total_images': len(results),
            'predictions_count': predictions,
            'predictions_percentage': {k: v / len(results) * 100 for k, v in predictions.items()},
            'average_confidence': {k: np.mean(v) for k, v in confidences.items()},
            'confidence_std': {k: np.std(v) for k, v in confidences.items()}
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary statistics saved to {summary_file}")

        # Print summary
        print(f"\n{'=' * 50}")
        print("BAYESIAN ATTRIBUTION SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total images: {summary['total_images']}")
        print("\nPredictions:")
        for gen, count in predictions.items():
            pct = count / len(results) * 100
            avg_conf = np.mean(confidences[gen])
            print(f"  {gen:20s}: {count:3d} ({pct:5.1f}%) - Avg confidence: {avg_conf:.3f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Bayesian Generator Attribution')
    parser.add_argument('--image', type=str, help='Single image path for attribution')
    parser.add_argument('--batch', action='store_true', help='Process batch of images')
    parser.add_argument('--input-dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-srec', action='store_true', help='Use uniform priors instead of SReC')

    args = parser.parse_args()

    # Create attributor
    attributor = BayesianAttributor(use_srec_priors=not args.no_srec)

    if args.image:
        # Single image attribution
        result = attributor.attribute_single_image(args.image)

        print(f"\n{'=' * 50}")
        print(f"ATTRIBUTION RESULT: {Path(args.image).name}")
        print(f"{'=' * 50}")
        print(f"Predicted Generator: {result['predicted_generator']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("\nPosterior Probabilities:")
        for gen, prob in sorted(result['posteriors'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {gen:20s}: {prob:.3f}")

    elif args.batch and args.input_dir:
        # Batch processing
        output_file = args.output if args.output else None
        results = attributor.attribute_batch(args.input_dir, output_file)

    else:
        print("Please specify either --image or --batch with --input-dir")
        parser.print_help()


if __name__ == "__main__":
    main()