#!/usr/bin/env python3
"""
Bayesian Generator Attribution

Compute Bayesian posteriors P(Generator|Image) using:
- Likelihoods from trained ResNet binary classifiers
- Priors from SReC D(l) values or RIGID scores (or uniform if not available)

Usage:
    python bayesian_attribution.py --image path/to/image.png --method srec --model imagenet
    python bayesian_attribution.py --batch --input-dir path/to/images/ --method rigid --model imagenet
    python bayesian_attribution.py --batch --input-dir path/to/images/ --method srec --model openimages
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
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base configuration
models_dir = '/mnt/ssd-data/vaidya/combined_pipeline/models/models_with_new_srec_weights/'
results_dir = '/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/bayesian_pipeline_results'
base_results_dir = '/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/priors'

# Only AI generators here. 'coco' is treated separately as the real class.
available_generators = [
    'dall-e2',
    'dall-e3',
    'firefly',
    'midjourneyV5',
    'midjourneyV6',
    'sdxl',
    'stable_diffusion_1-5',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class BayesianAttributor:
    """Bayesian generator attribution using binary classifiers and priors from different methods."""

    def __init__(self, method='srec', model='imagenet', use_priors=True, prior_temperature=1.0,
                 targeted_temperature=True):
        """
        Initialize Bayesian Attributor

        Args:
            method: 'srec' or 'rigid' or 'aeroblade' - method to use for computing priors
            model: 'imagenet' or 'openimages' - model to use (only for srec)
            use_priors: Whether to use method-based priors or uniform priors
            prior_temperature: Temperature for sharpening/smoothing priors (higher = sharper discrimination)
            targeted_temperature: If True, only apply temperature to similar generators; if False, apply to all
        """
        self.method = method.lower()
        self.model = model.lower()
        self.use_priors = use_priors
        self.prior_temperature = prior_temperature
        self.targeted_temperature = targeted_temperature

        # Set results directory based on method and model
        if self.method == 'srec':
            model_dir = 'IMAGENET_MODEL' if self.model == 'imagenet' else 'OPENIMAGES_MODEL'
            self.prior_results_dir = f'{base_results_dir}/SReC'
        elif self.method == 'rigid':
            self.prior_results_dir = f'{base_results_dir}/RIGID'
        elif self.method == 'aeroblade':
            self.prior_results_dir = f'{base_results_dir}/AEROBLADE'
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'srec', 'rigid', or 'aeroblade'")

        self.classifiers = self.load_binary_classifiers()
        self.priors = self.load_priors()

        logger.info(f"Loaded {len(self.classifiers)} binary classifiers")
        logger.info(f"Method: {self.method.upper()}, Model: {self.model.upper() if self.method == 'srec' else 'N/A'}")
        logger.info(f"Using {'method-based' if use_priors else 'uniform'} priors")
        logger.info(
            f"Prior temperature: {self.prior_temperature:.2f} ({'targeted' if self.targeted_temperature else 'global'})")

    def apply_temperature_to_priors(self, priors):
        """Apply targeted or global temperature scaling to priors for enhanced discrimination.

        If targeted_temperature=True: Only applies temperature when similar priors are detected.
        If targeted_temperature=False: Applies temperature to all priors.

        Higher temperature (>1.0) makes differences more pronounced (sharper).
        Lower temperature (<1.0) makes differences less pronounced (smoother).
        Temperature = 1.0 leaves priors unchanged.
        """
        if self.prior_temperature == 1.0:
            return priors

        if not self.targeted_temperature:
            # Global temperature scaling - apply to all priors
            logger.info(f"Applying global temperature {self.prior_temperature:.2f} to all priors")

            eps = 1e-12
            log_priors = {gen: np.log(prior + eps) for gen, prior in priors.items()}

            # Apply temperature scaling to all
            temp_log_priors = {gen: log_prior / self.prior_temperature
                               for gen, log_prior in log_priors.items()}

            # Convert back to probabilities and renormalize
            max_log = max(temp_log_priors.values())
            exp_priors = {gen: np.exp(log_prior - max_log)
                          for gen, log_prior in temp_log_priors.items()}

            total = sum(exp_priors.values())
            temperature_priors = {gen: exp_prior / total
                                  for gen, exp_prior in exp_priors.items()}

            logger.info(f"Global temperature scaling results:")
            for gen, prior in sorted(temperature_priors.items(), key=lambda x: x[1], reverse=True):
                original = priors[gen]
                change = prior - original
                logger.info(f"  {gen:20s}: {original:.6f} → {prior:.6f} ({change:+.6f})")

            return temperature_priors

        # Identify problematic generator pairs with very similar priors
        similar_threshold = 0.01  # 1% difference threshold
        problematic_pairs = []

        # Check for MidjourneyV5/V6 specifically (known issue)
        if 'midjourneyV5' in priors and 'midjourneyV6' in priors:
            v5_prior = priors['midjourneyV5']
            v6_prior = priors['midjourneyV6']
            difference = abs(v5_prior - v6_prior)

            if difference < similar_threshold:
                problematic_pairs.append(('midjourneyV5', 'midjourneyV6'))
                logger.info(
                    f"Detected similar MidjourneyV5/V6 priors: {v5_prior:.6f} vs {v6_prior:.6f} (diff: {difference:.6f})")

        # Check for other similar generator pairs
        gen_list = [g for g in priors.keys() if g != 'coco']
        for i, gen1 in enumerate(gen_list):
            for gen2 in gen_list[i + 1:]:
                if (gen1, gen2) not in problematic_pairs and (gen2, gen1) not in problematic_pairs:
                    diff = abs(priors[gen1] - priors[gen2])
                    if diff < similar_threshold:
                        problematic_pairs.append((gen1, gen2))
                        logger.info(
                            f"Detected similar priors: {gen1} ({priors[gen1]:.6f}) vs {gen2} ({priors[gen2]:.6f})")

        if not problematic_pairs:
            logger.info(
                f"No similar priors detected (threshold: {similar_threshold:.3f}). Skipping temperature scaling.")
            return priors

        logger.info(
            f"Applying temperature {self.prior_temperature:.2f} to enhance discrimination between similar generators")

        # Apply temperature only to the detected problematic generators
        problematic_gens = set()
        for pair in problematic_pairs:
            problematic_gens.update(pair)

        # Convert to log probabilities, apply temperature selectively
        eps = 1e-12
        log_priors = {gen: np.log(prior + eps) for gen, prior in priors.items()}

        # Apply temperature scaling only to problematic generators
        temp_log_priors = {}
        for gen, log_prior in log_priors.items():
            if gen in problematic_gens:
                temp_log_priors[gen] = log_prior / self.prior_temperature
            else:
                temp_log_priors[gen] = log_prior  # Keep unchanged

        # Convert back to probabilities and renormalize
        max_log = max(temp_log_priors.values())
        exp_priors = {gen: np.exp(log_prior - max_log)
                      for gen, log_prior in temp_log_priors.items()}

        total = sum(exp_priors.values())
        temperature_priors = {gen: exp_prior / total
                              for gen, exp_prior in exp_priors.items()}

        logger.info(f"Temperature scaling results:")
        for gen, prior in sorted(temperature_priors.items(), key=lambda x: x[1], reverse=True):
            original = priors[gen]
            change = prior - original
            status = "ENHANCED" if gen in problematic_gens else "unchanged"
            logger.info(f"  {gen:20s}: {original:.6f} → {prior:.6f} ({change:+.6f}) [{status}]")

        # Verify that problematic pairs now have better separation
        for gen1, gen2 in problematic_pairs:
            new_diff = abs(temperature_priors[gen1] - temperature_priors[gen2])
            old_diff = abs(priors[gen1] - priors[gen2])
            improvement = (new_diff - old_diff) / old_diff * 100 if old_diff > 0 else 0
            logger.info(f"  {gen1}/{gen2} separation: {old_diff:.6f} → {new_diff:.6f} (+{improvement:.1f}%)")

        return temperature_priors

    def load_binary_classifiers(self):
        """Load OvR binary classifiers trained with a single-logit BCE head.

        Expected filenames in models_dir:
          resnet50_dalle2_vs_nondalle2.pth
          resnet50_dalle3_vs_nondalle3.pth
          resnet50_firefly_vs_nonfirefly.pth
          resnet50_midjourneyv5_vs_nonmidjourneyv5.pth
          resnet50_midjourneyv6_vs_nonmidjourneyv6.pth
          resnet50_sdxl_vs_nonsdxl.pth
          resnet50_stablediffusion15_vs_nonstablediffusion15.pth
        """
        from pathlib import Path

        classifiers = {}

        # Map your logical generator names to the sanitized bases used in training
        name_to_base = {
            'dall-e2': 'dalle2',
            'dall-e3': 'dalle3',
            'firefly': 'firefly',
            'midjourneyV5': 'midjourneyv5',
            'midjourneyV6': 'midjourneyv6',
            'sdxl': 'sdxl',
            'stable_diffusion_1-5': 'stablediffusion15',
        }

        for generator in available_generators:
            base = name_to_base[generator]
            model_path = Path(models_dir) / f"resnet50_{base}_vs_non{base}.pth"

            if not model_path.exists():
                logger.warning(f"Model not found for {generator}: {model_path}")
                continue

            # Same architecture as in training: ResNet50 + Linear(1) head
            try:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            except Exception:
                model = models.resnet50(pretrained=True)

            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 1)

            # You saved a plain state_dict (not a checkpoint dict)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            model = model.to(device)
            model.eval()

            classifiers[generator] = model
            logger.info(f"Loaded OvR ResNet50 classifier for {generator} from {model_path}")

        return classifiers

    def load_priors(self):
        """Load or create priors P(Generator)."""
        if self.use_priors:
            # Strict mode: attempt to load method-based priors and fail loudly
            if self.method == 'srec':
                priors = self.load_srec_priors()
                logger.info(f"Loaded SReC-based priors ({self.model.upper()} model)")
            elif self.method == 'rigid':
                priors = self.load_rigid_priors()
                logger.info("Loaded RIGID-based priors")
            elif self.method == 'aeroblade':
                priors = self.load_aeroblade_priors()
                logger.info("Loaded Aeroblade-based priors")

            # Apply temperature scaling to enhance discrimination
            priors = self.apply_temperature_to_priors(priors)
            return priors

        # If priors are explicitly disabled, return uniform priors
        uniform_prior = 1.0 / (len(available_generators) + 1)  # +1 for coco
        priors = {gen: uniform_prior for gen in available_generators}
        priors['coco'] = uniform_prior  # Add real class

        logger.info("Using uniform priors (explicit --uniform flag)")
        return priors

    def load_srec_priors(self):
        """Load SReC D(l) values and convert to priors."""
        # Map JSON filenames to our generator names
        file_mapping = {
            'coco_d0.json': 'coco',
            'dall-e2_d0.json': 'dall-e2',
            'dall-e3_d0.json': 'dall-e3',
            'firefly_d0.json': 'firefly',
            'midjourneyV5_d0.json': 'midjourneyV5',
            'midjourneyV6_d0.json': 'midjourneyV6',
            'sdxl_d0.json': 'sdxl',
            'stable_diffusion_1-5_d0.json': 'stable_diffusion_1-5'
        }

        # Load D(l) values for each dataset
        dl_values = {}
        logger.info("Loading SReC D(l) values from individual JSON files:")
        missing = []
        for json_filename in file_mapping.keys():
            json_path = Path(self.prior_results_dir) / json_filename
            if not json_path.exists():
                missing.append(str(json_path))

        if missing:
            raise FileNotFoundError(f"SReC prior files missing: {missing}")

        for json_filename, generator in file_mapping.items():
            json_path = Path(self.prior_results_dir) / json_filename
            try:
                with open(json_path, 'r') as f:
                    image_dl_data = json.load(f)

                # Compute mean D(l) across all images in this dataset
                dl_list = [v for v in image_dl_data.values() if isinstance(v, (int, float))]
                if not dl_list:
                    raise ValueError(f"SReC file {json_path} contains no numeric D(l) values")

                mean_dl = float(np.mean(dl_list))
                std_dl = float(np.std(dl_list))

                dl_values[generator] = mean_dl
                logger.info(f"  {generator:20s}: D(l) = {mean_dl:7.4f} ± {std_dl:6.4f} ({len(dl_list):4d} images)")

            except Exception as e:
                raise RuntimeError(f"Error loading SReC priors from {json_path}: {e}")

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

    def load_rigid_priors(self):
        """Load RIGID scores and convert to priors."""
        # Map JSON filenames to our generator names
        file_mapping = {
            'coco_rigid_results.json': 'coco',
            'dall-e2_rigid_results.json': 'dall-e2',
            'dall-e3_rigid_results.json': 'dall-e3',
            'firefly_rigid_results.json': 'firefly',
            'midjourneyV5_rigid_results.json': 'midjourneyV5',  # <- capital V
            'midjourneyV6_rigid_results.json': 'midjourneyV6',  # <- capital V
            'sdxl_rigid_results.json': 'sdxl',
            'stable_diffusion_1-5_rigid_results.json': 'stable_diffusion_1-5'
        }

        # Load RIGID scores for each dataset
        rigid_scores = {}
        logger.info("Loading RIGID scores from individual JSON files:")

        # Ensure all required RIGID JSON files exist
        missing = []
        for json_filename in file_mapping.keys():
            json_path = Path(self.prior_results_dir) / json_filename
            if not json_path.exists():
                missing.append(str(json_path))

        if missing:
            raise FileNotFoundError(f"RIGID prior files missing: {missing}")

        for json_filename, generator in file_mapping.items():
            json_path = Path(self.prior_results_dir) / json_filename
            try:
                with open(json_path, 'r') as f:
                    image_rigid_data = json.load(f)

                # Compute mean RIGID score across all images in this dataset
                score_list = [v for v in image_rigid_data.values() if isinstance(v, (int, float))]
                if not score_list:
                    raise ValueError(f"RIGID file {json_path} contains no numeric scores")

                mean_score = float(np.mean(score_list))
                std_score = float(np.std(score_list))

                rigid_scores[generator] = mean_score
                logger.info(
                    f"  {generator:20s}: RIGID = {mean_score:7.4f} ± {std_score:6.4f} ({len(score_list):4d} images)")

            except Exception as e:
                raise RuntimeError(f"Error loading RIGID priors from {json_path}: {e}")

        # Convert RIGID scores to priors
        # Higher score = more likely to be AI-generated
        # Lower score = more likely to be real

        # Separate real and AI generators
        coco_score = rigid_scores['coco']
        ai_generators = [gen for gen in available_generators]
        ai_scores = [rigid_scores[gen] for gen in ai_generators]

        logger.info(f"\nRIGID score statistics:")
        logger.info(f"  Real (COCO):     {coco_score:7.4f}")
        logger.info(f"  AI generators:   {np.mean(ai_scores):7.4f} ± {np.std(ai_scores):6.4f}")

        # Convert RIGID scores to priors using a simpler approach
        # Since all scores are very high (>0.95) and similar, use differences for discrimination

        # Calculate relative scores (difference from minimum)
        all_scores = [coco_score] + ai_scores
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score

        # Normalize scores to [0, 1] range based on relative differences
        if score_range > 0:
            # Use differences from minimum to create discrimination
            coco_norm = (coco_score - min_score) / score_range
            ai_norm = [(score - min_score) / score_range for score in ai_scores]
        else:
            # All scores are identical - cannot form informative priors
            raise ValueError("Degenerate RIGID scores: all scores identical; cannot derive priors")

        # Use gentler temperature for normalized scores
        temperature = 0.5

        # Compute priors with minimum threshold to prevent zero priors
        min_prior = 0.01  # Minimum 1% prior for any generator

        # For AI generators: use normalized scores directly
        ai_norm = np.array(ai_norm)
        ai_norm_centered = ai_norm - np.mean(ai_norm)
        # clip to a small range so no one gets a crazy advantage
        ai_norm_centered = np.clip(ai_norm_centered, -0.1, 0.1)
        ai_exp_vals = np.exp(ai_norm_centered * temperature)
        ai_sum = np.sum(ai_exp_vals)

        # For COCO: invert the normalized score (lower RIGID = more likely real)
        # But since COCO has high RIGID score (wrongly classified as AI), give it a baseline prior
        real_exp = np.exp((1.0 - coco_norm) * temperature)

        # Balance between real and AI priors
        total_strength = real_exp + ai_sum
        raw_real_prior = real_exp / total_strength
        raw_ai_priors = ai_exp_vals / ai_sum * (1.0 - raw_real_prior)

        # Apply minimum prior threshold
        priors = {}
        priors['coco'] = max(raw_real_prior, min_prior)

        for i, generator in enumerate(ai_generators):
            priors[generator] = max(raw_ai_priors[i], min_prior)

        # Renormalize to sum to 1
        total_prior = sum(priors.values())
        for gen in priors:
            priors[gen] /= total_prior

        logger.info("\nConverted to priors:")
        for gen, prior in sorted(priors.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {gen:20s}: P = {prior:6.3f}")

        return priors

    def load_aeroblade_priors(self):
        """Load Aeroblade LPIPS distances and convert to priors using the
        same high-level conversion logic as `load_srec_priors` and
        `load_rigid_priors`.

        Steps:
        - Read per-repo JSON files under `self.prior_results_dir` containing
          mappings image_path -> LPIPS distance.
        - Compute mean LPIPS distance per repo (generator).
        - Convert mean distances to priors: for AI generators we use
          ai_exp_vals = exp(-alpha * mean_distance) (smaller distance => larger)
          and for COCO (real) we compute real_strength = exp(-alpha * coco_mean).
        - Balance real vs AI priors and renormalize so priors sum to 1.
        """
        json_dir = Path(self.prior_results_dir)
        gens_all = available_generators + ['coco']

        if not json_dir.exists():
            raise FileNotFoundError(f"Aeroblade prior directory not found: {json_dir}")

        json_files = list(json_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No Aeroblade JSON files found in {json_dir}")

        # Load mean LPIPS distance per generator
        mean_dist = {}
        for jf in json_files:
            name = jf.stem.lower()
            matched = None
            for g in gens_all:
                if g.lower() in name:
                    matched = g
                    break
            if matched is None:
                # try simple prefix match
                parts = name.split('_')
                if parts and parts[0] in [g.lower() for g in gens_all]:
                    matched = parts[0]
                else:
                    logger.debug(f"Skipping unrecognized Aeroblade file: {jf}")
                    continue

            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                vals = [v for v in data.values() if isinstance(v, (int, float))]
                if not vals:
                    raise ValueError(f"Aeroblade file {jf} contains no numeric LPIPS values")
                mean_dist[matched] = float(np.mean(vals))
                logger.info(f"Aeroblade: {matched:20s} mean_lpips = {mean_dist[matched]:.6f} ({len(vals)} images)")
            except Exception as e:
                raise RuntimeError(f"Failed to load Aeroblade file {jf}: {e}")

        # Fill missing generators with max observed distance (least AI-like)
        if mean_dist:
            observed = list(mean_dist.values())
            max_obs = max(observed)
        else:
            max_obs = 1.0

        ai_generators = [g for g in available_generators]
        ai_dist_values = [mean_dist.get(g, max_obs) for g in ai_generators]
        coco_dist = mean_dist.get('coco', max_obs)

        logger.info(f"\nAeroblade D(distance) statistics:")
        logger.info(f"  Real (COCO):     {coco_dist:7.6f}")
        logger.info(f"  AI generators:   {np.mean(ai_dist_values):7.6f} ± {np.std(ai_dist_values):7.6f}")

        # Convert distances to priors using softmax-like mapping
        alpha = 1.0
        ai_arr = np.array(ai_dist_values)
        ai_exp_vals = np.exp(-alpha * ai_arr)
        ai_exp_sum = np.sum(ai_exp_vals)

        priors = {}
        if ai_exp_sum == 0:
            raise ValueError("Degenerate Aeroblade scores: exponential sum is zero; cannot form priors")

        for i, generator in enumerate(ai_generators):
            priors[generator] = ai_exp_vals[i] / ai_exp_sum

        # For real: inverse relationship (lower distance -> more likely real)
        real_strength = np.exp(-alpha * coco_dist)
        ai_total_strength = np.sum(ai_exp_vals)

        total_strength = real_strength + ai_total_strength
        real_prior = real_strength / total_strength
        ai_scale = (1.0 - real_prior)

        priors['coco'] = real_prior
        for gen in ai_generators:
            priors[gen] *= ai_scale

        # Normalize to sum to 1 (numerical safety)
        total_prior = sum(priors.values())
        for gen in priors:
            priors[gen] /= total_prior

        logger.info("\nConverted Aeroblade mean distances to priors:")
        for gen, prior in sorted(priors.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {gen:20s}: P = {prior:6.3f}")

        return priors

    def load_aeroblade_per_image(self):
        """Load per-image Aeroblade distances from priors/AEROBLADE/*.json.
        Builds a mapping: self.aeroblade_by_image[image_path] = {generator: distance}
        """
        if hasattr(self, 'aeroblade_by_image') and self.aeroblade_by_image is not None:
            return

        self.aeroblade_by_image = {}
        folder = Path(base_results_dir) / 'AEROBLADE'
        if not folder.exists():
            logger.info(f"Aeroblade folder not found: {folder}")
            return

        # Iterate JSON files and infer generator name from filename
        for jf in folder.glob('*.json'):
            try:
                data = json.load(open(jf))
            except Exception as e:
                logger.warning(f"Failed to load Aeroblade JSON {jf}: {e}")
                continue

            # infer generator key from filename by matching known generators
            fname = jf.name.lower()
            gen_key = None
            for g in available_generators + ['coco']:
                if g.lower() in fname:
                    gen_key = g
                    break
            if gen_key is None:
                # fallback: strip extensions
                gen_key = jf.stem

            # data expected to be mapping image_filename -> distance
            for img_name, dist in data.items():
                # construct full path as stored in results (dataset_512/...)
                # if img_name already contains path, use it as-is
                if img_name.startswith('/'):
                    img_path = img_name
                else:
                    # try to find which dataset folder it belongs to by scanning
                    # common prefix used in results: /mnt/hdd-data/vaidya/dataset_512
                    img_path = str(Path('/mnt/hdd-data/vaidya/dataset_512') / img_name)

                rec = self.aeroblade_by_image.get(img_path, {})
                rec[gen_key] = float(dist)
                self.aeroblade_by_image[img_path] = rec

        logger.info(f"Loaded Aeroblade per-image distances for {len(self.aeroblade_by_image)} images")

    def get_image_priors(self, image_path, method_pref='aeroblade'):
        """Return a prior dict for this image. If per-image Aeroblade data exists, use it.
        Otherwise fallback to dataset-level priors stored in self.priors.
        """
        # Try per-image Aeroblade priors first
        try:
            self.load_aeroblade_per_image()
        except Exception:
            pass

        if hasattr(self, 'aeroblade_by_image') and self.aeroblade_by_image:
            img_key = str(image_path)
            if img_key in self.aeroblade_by_image:
                distances = self.aeroblade_by_image[img_key]
                # ensure all generators present
                gens = available_generators.copy()
                # build vector of distances; missing => large value
                dlist = [distances.get(g, 10.0) for g in gens]
                # convert to scores: smaller distance -> larger score
                alpha = 10.0
                scores = np.exp(-alpha * np.array(dlist))
                # include coco as small baseline unless present
                coco_score = distances.get('coco', 1e-6)
                # Normalize across AI generators and add coco minimal score
                total = scores.sum() + coco_score
                priors = {g: float(scores[i] / total) for i, g in enumerate(gens)}
                priors['coco'] = float(coco_score / total)
                return priors

        # Fallback: return dataset-level priors
        return self.priors

    def get_likelihoods(self, image_path):
        """Get P(Image|Generator) from OvR binary classifiers.

        For each generator g:
          model_g outputs a single logit -> sigmoid = P(class = g).
        For real (coco):
          P(Image|Real) is the geometric mean of P(non-g) = 1 - P(g) over all generators.
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        likelihoods = {}
        real_components = []

        with torch.no_grad():
            for generator, model in self.classifiers.items():
                try:
                    logit = model(image_tensor).squeeze(1)  # shape (1,)
                    p_gen = torch.sigmoid(logit).item()     # P(class = generator)
                    likelihoods[generator] = p_gen

                    # contribution to "real" likelihood: P(non-generator) = 1 - p_gen
                    real_components.append(max(1.0 - p_gen, 1e-6))

                    logger.debug(f"{generator}: P(generator) = {p_gen:.4f}")

                except Exception as e:
                    logger.warning(f"Error with {generator} classifier: {e}")
                    likelihoods[generator] = 0.5
                    real_components.append(0.5)

        # Geometric mean for P(Image|Real = coco)
        if real_components:
            real_likelihood = float(np.prod(real_components) ** (1.0 / len(real_components)))
        else:
            real_likelihood = 0.5

        likelihoods['coco'] = real_likelihood
        logger.debug(f"Combined P(Real = coco) = {real_likelihood:.4f}")

        return likelihoods

    def compute_posteriors(self, image_path):
        """Compute Bayesian posteriors P(Generator|Image)."""
        # Get likelihoods P(Image|Generator)
        likelihoods = self.get_likelihoods(image_path)
        # Get per-image priors if available (falls back to dataset priors)
        priors = self.get_image_priors(image_path)

        # Compute unnormalized posteriors: P(Generator|Image) ∝ P(Image|Generator) × P(Generator)
        log_posteriors = {}
        eps = 1e-12  # Numerical stability

        for generator in available_generators + ['coco']:
            likelihood = likelihoods.get(generator, eps)
            prior = priors.get(generator, eps)

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
            'priors': priors,
            'posteriors': posteriors,
            'predicted_generator': max(posteriors.items(), key=lambda x: x[1])[0],
            'confidence': max(posteriors.values())
        }

    def attribute_single_image(self, image_path):
        """Attribution for a single image."""
        logger.info(f"Attributing: {image_path}")
        return self.compute_posteriors(image_path)

    def attribute_batch(self, input_dir, output_file=None, max_images=None):
        """Attribution for all images in a directory."""
        input_path = Path(input_dir)
        # Search recursively for common image extensions so users can pass a
        # dataset root containing per-generator subfolders (e.g. dataset_512/)
        exts = {'.png', '.jpg', '.jpeg'}
        image_files = [p for p in input_path.rglob('*') if p.suffix.lower() in exts]
        if max_images is not None:
            image_files = image_files[:max_images]

        if not image_files:
            logger.error(f"No images found in {input_dir} (searched recursively)")
            return None

        logger.info(f"Processing {len(image_files)} images...")

        results = []
        for img_file in image_files:
            try:
                result = self.compute_posteriors(img_file)
                logger.debug(f"Result for {img_file}: {result}")
                results.append(result)

                if len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)}/{len(image_files)} images")

            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                traceback.print_exc()
                continue

        # Save results
        if output_file is None:
            output_file = Path(results_dir) / "bayesian_attribution_results.json"
        output_path = Path(output_file) if output_file is not None else Path(output_file)
        # ensure parent dir exists
        output_path = Path(output_file) if output_file is not None else Path(
            results_dir) / "bayesian_attribution_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved {len(results)} attribution results to {output_path}")

        # Generate summary statistics
        self.generate_summary(results, Path(output_file).parent / "attribution_summary.json")

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
    parser.add_argument('--method', type=str, default='srec', choices=['srec', 'rigid', 'aeroblade'],
                        help='Method for computing priors: srec, rigid, or aeroblade (default: srec)')
    parser.add_argument('--model', type=str, default='imagenet', choices=['imagenet', 'openimages'],
                        help='Model to use for SREC priors: imagenet or openimages (default: imagenet)')
    parser.add_argument('--uniform', action='store_true',
                        help='Use uniform priors instead of method-based priors')
    parser.add_argument('--prior-temperature', type=float, default=1.0,
                        help='Temperature for prior sharpening/smoothing (default: 1.0, higher=sharper discrimination)')
    parser.add_argument('--global-temperature', action='store_true',
                        help='Apply temperature to all priors instead of only similar ones')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process in batch mode')

    args = parser.parse_args()

    # Create attributor
    attributor = BayesianAttributor(
        method=args.method,
        model=args.model,
        use_priors=not args.uniform,
        prior_temperature=args.prior_temperature,
        targeted_temperature=not args.global_temperature
    )

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
        max_images = args.max_images if hasattr(args, 'max_images') else None
        results = attributor.attribute_batch(args.input_dir, output_file, max_images=max_images)

    else:
        print("Please specify either --image or --batch with --input-dir")
        parser.print_help()


if __name__ == "__main__":
    main()