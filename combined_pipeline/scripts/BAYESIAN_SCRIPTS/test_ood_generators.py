#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Generator Attribution Test

Tests whether unknown AI generators are correctly:
1. NOT misclassified as Real
2. Assigned to one of the known generator classes (fallback behavior)

Tested with generators not seen during training:
- Hotpot AI
- Other unknown generators from "In the Wild" dataset

Usage:
    python test_ood_generators.py \
        --input-dir /path/to/ood/images \
        --generator-name hotpot-ai \
        --method srec \
        --model openimages \
        --output-dir /path/to/results
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
import argparse
import logging
from PIL import Image
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Optional Bayesian attributor (for posterior-based assignment)
from combined_pipeline.scripts.BAYESIAN_SCRIPTS.bayesian_attribution import BayesianAttributor
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODELS_DIR = '/mnt/ssd-data/vaidya/combined_pipeline/models/'
TRAINED_GENERATORS = [
    'dalle3',
    'firefly',
    'midjourneyV5',
    'midjourneyV6',
    'sdxl',
    'stable_diffusion_1_5',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class OODGeneratorTester:
    """Test OOD generator attribution behavior."""

    def __init__(self, method='srec', model='openimages', use_bayesian=False, bayes_threshold=0.5):
        """Initialize tester with trained classifiers.

        Args:
            use_bayesian: If True, use `BayesianAttributor` posteriors for assignment.
            bayes_threshold: Posterior confidence threshold for accepting Bayesian prediction.
        """
        self.method = method
        self.model = model
        self.use_bayesian = use_bayesian
        self.bayes_threshold = bayes_threshold

        self.classifiers = self.load_classifiers()
        logger.info(f"Loaded {len(self.classifiers)} binary classifiers for OOD testing")

        self.bayesian = None
        if self.use_bayesian:
            # Instantiate BayesianAttributor with matching method/model choices
            try:
                self.bayesian = BayesianAttributor(method=self.method, model=self.model)
                logger.info("Initialized BayesianAttributor for posterior-based assignments")
            except Exception as e:
                logger.warning(f"Failed to initialize BayesianAttributor: {e}")
                self.bayesian = None

    def load_classifiers(self):
        """Load trained binary classifiers."""
        classifiers = {}
        
        for gen in TRAINED_GENERATORS:
            # Try ResNet50 first
            model_path = Path(MODELS_DIR) / f"resnet50_{gen}_vs_coco.pth"
            arch = 'resnet50'
            
            # Try ResNet18 if ResNet50 not found
            if not model_path.exists():
                model_path = Path(MODELS_DIR) / f"resnet18_{gen}_vs_coco.pth"
                arch = 'resnet18'
            
            if not model_path.exists():
                logger.warning(f"Classifier not found for {gen}")
                continue
            
            # Load model
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 2)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            classifiers[gen] = model
            logger.info(f"Loaded classifier for {gen}")
        
        return classifiers

    def get_classifier_probabilities(self, image_path):
        """Get P(Generator|Image) from all binary classifiers.
        
        Each classifier outputs P(Generator) vs P(Real).
        We extract P(Generator) as the probability the image is from that generator.
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            probabilities = {}
            
            with torch.no_grad():
                for gen_name, classifier in self.classifiers.items():
                    logits = classifier(img_tensor)
                    probs = torch.softmax(logits, dim=1)
                    
                    # prob_fake = P(Generator) = softmax output for class 1
                    prob_generator = probs[0, 1].item()
                    prob_real = probs[0, 0].item()
                    
                    probabilities[gen_name] = {
                        'prob_generator': prob_generator,
                        'prob_real': prob_real,
                        'confidence': max(prob_generator, prob_real)
                    }
            
            return probabilities
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def test_ood_image(self, image_path):
        """Test a single OOD image and return classification results."""
        logger.info(f"Testing OOD image: {image_path}")
        
        probs = self.get_classifier_probabilities(image_path)
        if probs is None:
            return None
        
        # Analysis
        result = {
            'image': Path(image_path).name,
            'path': str(image_path),
            'classifier_outputs': probs,
        }
        
        # Key metrics for OOD testing
        avg_prob_generator = np.mean([p['prob_generator'] for p in probs.values()])
        avg_prob_real = np.mean([p['prob_real'] for p in probs.values()])
        
        result['metrics'] = {
            'avg_prob_generator': avg_prob_generator,
            'avg_prob_real': avg_prob_real,
            'avg_confidence': np.mean([p['confidence'] for p in probs.values()]),
            'max_generator_prob': max([p['prob_generator'] for p in probs.values()]),
            'min_generator_prob': min([p['prob_generator'] for p in probs.values()]),
        }
        
        # Top predicted generator
        top_gen = max(probs.items(), key=lambda x: x[1]['prob_generator'])
        result['top_predicted_generator'] = top_gen[0]
        result['top_generator_prob'] = top_gen[1]['prob_generator']

        # Default heuristic assigned label (kept for compatibility)
        if avg_prob_real >= avg_prob_generator:
            heuristic_label = 'Real'
        else:
            heuristic_label = result['top_predicted_generator']

        result['assigned_label'] = heuristic_label

        # If Bayesian option enabled and available, compute posteriors and
        # use them to override the assigned label based on `bayes_threshold`.
        if self.use_bayesian and self.bayesian is not None:
            try:
                bayes_res = self.bayesian.attribute_single_image(image_path)
                post = bayes_res.get('posteriors', {})
                predicted = bayes_res.get('predicted_generator')
                confidence = bayes_res.get('confidence', 0.0)

                result['bayesian_posteriors'] = post
                result['bayesian_predicted'] = predicted
                result['bayesian_confidence'] = float(confidence)

                # Map Bayesian prediction to assigned label semantics
                if predicted == 'coco':
                    bayes_label = 'Real'
                else:
                    # Accept Bayesian prediction only if confidence >= threshold
                    bayes_label = predicted if confidence >= self.bayes_threshold else 'Unknown'

                result['bayesian_assigned_label'] = bayes_label

                # Override assigned_label with bayesian assignment
                result['assigned_label'] = bayes_label

                # Add bayesian critical checks (mirrors heuristic checks)
                result['bayesian_checks'] = {
                    'not_classified_as_real': bool(post.get('coco', 0.0) < self.bayes_threshold),
                    'assigned_to_generator': bool(predicted != 'coco' and confidence >= self.bayes_threshold)
                }
            except Exception as e:
                logger.warning(f"Bayesian attribution failed for {image_path}: {e}")
        
        # Check for critical failures
        result['critical_checks'] = {
            'not_classified_as_real': bool(avg_prob_real < 0.5),  # Should not think it's real
            'recognized_as_fake': bool(avg_prob_generator > 0.5),  # Should recognize as fake
            'assigned_to_generator': bool(result['top_generator_prob'] > 0.3),  # Falls back to a generator
        }
        
        return result

    def test_directory(self, image_dir, output_dir=None, max_images=None):
        """Test all images in a directory."""
        image_dir = Path(image_dir)
        images = list(image_dir.rglob('*.png')) + list(image_dir.rglob('*.jpg')) + list(image_dir.rglob('*.jpeg'))
        
        logger.info(f"Found {len(images)} images to test in {image_dir}")
        
        if max_images and len(images) > max_images:
            images = random.sample(images, max_images)
            logger.info(f"Sampled {max_images} images for testing")
        
        results = []
        for i, img_path in enumerate(images, 1):
            logger.info(f"[{i}/{len(images)}] Processing {img_path.name}")
            result = self.test_ood_image(str(img_path))
            if result:
                results.append(result)
        
        return self.analyze_results(results, output_dir)

    def analyze_results(self, results, output_dir=None):
        """Analyze and summarize OOD test results."""
        if not results:
            logger.warning("No results to analyze")
            return None
        
        logger.info("\n" + "="*70)
        logger.info("OOD GENERATOR ATTRIBUTION TEST RESULTS")
        logger.info("="*70)
        
        # Summary statistics
        summary = {
            'total_images_tested': len(results),
            'images_not_classified_as_real': sum(1 for r in results if r['critical_checks']['not_classified_as_real']),
            'images_recognized_as_fake': sum(1 for r in results if r['critical_checks']['recognized_as_fake']),
            'images_assigned_to_generator': sum(1 for r in results if r['critical_checks']['assigned_to_generator']),
        }
        
        logger.info(f"\nTest Summary:")
        logger.info(f"  Total images tested: {summary['total_images_tested']}")
        logger.info(f"  ✓ NOT misclassified as Real: {summary['images_not_classified_as_real']}/{summary['total_images_tested']} ({100*summary['images_not_classified_as_real']/summary['total_images_tested']:.1f}%)")
        logger.info(f"  ✓ Recognized as fake: {summary['images_recognized_as_fake']}/{summary['total_images_tested']} ({100*summary['images_recognized_as_fake']/summary['total_images_tested']:.1f}%)")
        logger.info(f"  ✓ Assigned to known generator: {summary['images_assigned_to_generator']}/{summary['total_images_tested']} ({100*summary['images_assigned_to_generator']/summary['total_images_tested']:.1f}%)")
        
        # Per-generator assignment frequency
        gen_assignments = defaultdict(int)
        for result in results:
            # Use the assigned_label which may be 'Real' or one of the generators
            label = result.get('assigned_label', result['top_predicted_generator'])
            gen_assignments[label] += 1
        
        logger.info(f"\nFallback Generator Assignments:")
        for gen in sorted(gen_assignments.keys()):
            count = gen_assignments[gen]
            logger.info(f"  {gen}: {count} images ({100*count/len(results):.1f}%)")
        
        # Average metrics
        avg_prob_gen = np.mean([r['metrics']['avg_prob_generator'] for r in results])
        avg_prob_real = np.mean([r['metrics']['avg_prob_real'] for r in results])
        
        logger.info(f"\nAverage Classifier Outputs:")
        logger.info(f"  Avg P(Generator): {avg_prob_gen:.3f}")
        logger.info(f"  Avg P(Real): {avg_prob_real:.3f}")
        logger.info(f"  Ratio (Generator/Real): {avg_prob_gen/max(avg_prob_real, 0.001):.2f}x")
        
        # Save detailed results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            results_json = {
                'summary': summary,
                'generator_assignments': dict(gen_assignments),
                'average_metrics': {
                    'avg_prob_generator': float(avg_prob_gen),
                    'avg_prob_real': float(avg_prob_real),
                    'ratio': float(avg_prob_gen / max(avg_prob_real, 0.001))
                },
                'detailed_results': results
            }
            
            json_path = output_dir / 'ood_test_results.json'
            with open(json_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            logger.info(f"\n✓ Detailed results saved to: {json_path}")
            
            # Create visualizations
            self.create_visualizations(results, output_dir)
            
            logger.info(f"✓ Results saved to: {output_dir}")
        
        logger.info("="*70)
        
        return {
            'summary': summary,
            'generator_assignments': dict(gen_assignments),
            'results': results
        }

    def create_visualizations(self, results, output_dir):
        """Create visualization plots for OOD test results."""
        output_dir = Path(output_dir)
        
        # 1. Distribution of P(Generator) vs P(Real)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        prob_gen = [r['metrics']['avg_prob_generator'] for r in results]
        prob_real = [r['metrics']['avg_prob_real'] for r in results]
        
        ax1.hist(prob_gen, bins=20, alpha=0.7, label='P(Generator)', color='red')
        ax1.hist(prob_real, bins=20, alpha=0.7, label='P(Real)', color='green')
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('OOD Images: Classifier Output Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Scatter plot
        ax2.scatter(prob_real, prob_gen, alpha=0.6, s=50)
        ax2.axhline(y=0.5, color='red', linestyle='--', label='P(Gen) = 0.5')
        ax2.axvline(x=0.5, color='green', linestyle='--', label='P(Real) = 0.5')
        ax2.set_xlabel('P(Real)')
        ax2.set_ylabel('P(Generator)')
        ax2.set_title('OOD Images: Real vs Generator Probability')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ood_probability_distribution.png', dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved probability distribution plot")
        plt.close()
        
        # 3. Generator assignment distribution
        gen_assignments = defaultdict(int)
        for result in results:
            # Use the assigned_label which may be 'Real' or one of the generators
            label = result.get('assigned_label', result['top_predicted_generator'])
            gen_assignments[label] += 1
        
        fig, ax = plt.subplots(figsize=(12, 6))
        gens = list(gen_assignments.keys())
        counts = list(gen_assignments.values())
        
        # Color Real differently from generators
        colors = ['green' if g == 'Real' else 'steelblue' for g in gens]
        
        ax.bar(gens, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Assigned Label', fontsize=12)
        ax.set_ylabel('Number of OOD Images', fontsize=12)
        ax.set_title('OOD Image Assignments: Real vs Generator Fallbacks', fontsize=14)
        ax.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ood_generator_assignments.png', dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved generator assignment plot")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test OOD generator attribution'
    )
    parser.add_argument('--input-dir', required=True, help='Directory with OOD images')
    parser.add_argument('--generator-name', default='unknown', help='Name of OOD generator (for logging)')
    parser.add_argument('--method', default='srec', choices=['srec', 'rigid', 'aeroblade'])
    parser.add_argument('--model', default='openimages', choices=['openimages', 'imagenet'])
    parser.add_argument('--output-dir', default=None, help='Directory to save results')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to test (random sample)')
    parser.add_argument('--use-bayesian', action='store_true', help='Use BayesianAttributor posteriors for assignment')
    parser.add_argument('--bayes-threshold', type=float, default=0.5, help='Posterior confidence threshold for Bayesian assignment')
    
    args = parser.parse_args()
    
    logger.info(f"Testing OOD Generator: {args.generator_name}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Method: {args.method}, Model: {args.model}")
    
    # Initialize tester
    tester = OODGeneratorTester(method=args.method, model=args.model,
                                use_bayesian=args.use_bayesian, bayes_threshold=args.bayes_threshold)
    
    # Run tests
    results = tester.test_directory(args.input_dir, output_dir=args.output_dir, max_images=args.max_images)
    
    if results and results['summary']['images_not_classified_as_real'] == results['summary']['total_images_tested']:
        logger.info("\n✓ SUCCESS: All OOD images correctly NOT classified as Real!")
    elif results and results['summary']['images_assigned_to_generator'] > 0:
        logger.info("\n⚠ PARTIAL SUCCESS: OOD images assigned to known generators (fallback behavior)")
    else:
        logger.warning("\n✗ FAILURE: Some OOD images may have been misclassified as Real")


if __name__ == '__main__':
    main()
