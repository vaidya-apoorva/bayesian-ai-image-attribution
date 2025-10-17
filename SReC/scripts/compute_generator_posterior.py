import os
import json
import joblib
import numpy as np
from collections import defaultdict
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "kde_dirs": {
        "coco": "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED/coco_trained_JPEG_80/joblib",
        "openimages": "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED/openimages_trained_JPEG_80/joblib",
    },
    "classifier_likelihoods_path": "/mnt/ssd-data/vaidya/SReC/results/classifier_result_updated.json",
    "image_gaps_path": "/mnt/ssd-data/vaidya/SReC/results/image_coding_cost_open_images.json",
    "output_dir": "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED_classifier_results/JPEG80",
    "beta": 3.0,
    "epsilon": 1e-10,
    "debug_samples": 5
}

# Generator → Coarse label mapping
GENERATOR_GROUPS = {
    "coco_d0": "Real",
    "raise_d0": "Real", 
    "dalle2_d0": "DALL-E",
    "dalle3_d0": "DALL-E",
    "midjourneyV5_d0": "MidJourney",
    "sdxl_d0": "StableDiffusion",
}

# Optimized ground truth inference with caching
_GT_CACHE = {}
def infer_ground_truth(path):
    """Optimized ground truth inference with caching"""
    if path in _GT_CACHE:
        return _GT_CACHE[path]
    
    path_lower = path.lower()
    if "coco" in path_lower or "raise" in path_lower:
        result = "Real"
    elif "dalle2" in path_lower or "dalle3" in path_lower:
        result = "DALL-E"
    elif "midjourney" in path_lower:
        result = "MidJourney"
    elif "sdxl" in path_lower:
        result = "StableDiffusion"
    else:
        result = None
    
    _GT_CACHE[path] = result
    return result

def load_data():
    """Load all required data files with error handling"""
    try:
        logger.info("Loading classifier likelihoods...")
        with open(CONFIG["classifier_likelihoods_path"]) as f:
            classifier_likelihoods = json.load(f)
        
        logger.info("Loading image gaps...")
        with open(CONFIG["image_gaps_path"]) as f:
            image_gaps = json.load(f)
            
        logger.info(f"Loaded {len(classifier_likelihoods)} classifier entries and {len(image_gaps)} gap entries")
        return classifier_likelihoods, image_gaps
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def load_kde_models(kde_dir):
    """Load KDE models from directory with optimized error handling"""
    kde_models = {}
    kde_path = Path(kde_dir)
    
    if not kde_path.exists():
        logger.warning(f"KDE directory does not exist: {kde_dir}")
        return kde_models
    
    for fname in kde_path.glob("kde_model_*.joblib"):
        gen = fname.stem[len("kde_model_"):]
        try:
            kde_models[gen] = joblib.load(fname)
        except Exception as e:
            logger.warning(f"Failed to load KDE model {fname}: {e}")
    
    return kde_models

def compute_prior_vectorized(gaps, kde_model, epsilon=1e-10):
    """Vectorized prior computation for better performance"""
    if kde_model is None or len(gaps) == 0:
        return np.full(len(gaps), epsilon)
    
    try:
        gaps_array = np.array(gaps).reshape(-1, 1)
        log_densities = kde_model.score_samples(gaps_array)
        return np.maximum(np.exp(log_densities), epsilon)
    except Exception as e:
        logger.warning(f"Error in vectorized prior computation: {e}")
        return np.full(len(gaps), epsilon)

def compute_posteriors_optimized(likelihoods, gaps, kde_models, beta=3.0):
    """Optimized posterior computation with batch processing"""
    posteriors = {}
    skipped = 0
    
    # Pre-filter valid images
    valid_images = []
    for img_path in likelihoods.keys():
        img_name = os.path.basename(img_path)
        entry = gaps.get(img_name)
        if entry is None:
            skipped += 1
            continue
        gap = entry if isinstance(entry, (int, float)) else entry.get("gap")
        if gap is None:
            skipped += 1
            continue
        valid_images.append((img_path, img_name, gap))
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} images due to missing gap data")
    
    logger.info(f"Processing {len(valid_images)} valid images...")
    
    # Pre-compute generator list
    valid_gens = [gen for gen in kde_models.keys() if gen in GENERATOR_GROUPS]
    
    for img_path, img_name, gap in valid_images:
        like_dict = likelihoods[img_path]
        
        log_unnorm = {}
        for gen in valid_gens:
            coarse_label = GENERATOR_GROUPS[gen]
            like = max(like_dict.get(coarse_label, 0.0), CONFIG["epsilon"])
            
            # Compute prior
            kde_model = kde_models.get(gen)
            if kde_model is None:
                prior = CONFIG["epsilon"]
            else:
                try:
                    log_density = kde_model.score_samples([[gap]])[0]
                    prior = max(np.exp(log_density), CONFIG["epsilon"])
                except:
                    prior = CONFIG["epsilon"]
            
            log_unnorm[gen] = np.log(like) + beta * np.log(prior)
        
        if not log_unnorm:
            continue
            
        # Normalize in log space for numerical stability
        max_log = max(log_unnorm.values())
        exp_unnorm = {gen: np.exp(val - max_log) for gen, val in log_unnorm.items()}
        total = sum(exp_unnorm.values()) or 1e-8
        posteriors[img_path] = {gen: val / total for gen, val in exp_unnorm.items()}
    
    return posteriors

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Load data once
    classifier_likelihoods, image_gaps = load_data()
    
    # Process each KDE directory
    for name, kde_dir in CONFIG["kde_dirs"].items():
        logger.info(f"Processing {name} dataset...")
        
        # Load KDE models
        kde_models = load_kde_models(kde_dir)
        logger.info(f"Loaded {len(kde_models)} KDE models for {name}: {sorted(kde_models.keys())}")
        
        if not kde_models:
            logger.warning(f"No KDE models found for {name}, skipping...")
            continue
        
        # Compute posteriors
        post = compute_posteriors_optimized(classifier_likelihoods, image_gaps, kde_models, CONFIG["beta"])
        
        # Debug output (limited for performance)
        if CONFIG["debug_samples"] > 0:
            logger.info(f"Sample Prior Influence for {name}:")
            for img_path in list(post)[:CONFIG["debug_samples"]]:
                img_name = os.path.basename(img_path)
                gap = image_gaps.get(img_name)
                if isinstance(gap, dict):
                    gap = gap.get("gap")
                logger.info(f"  {img_name} | Gap: {gap}")
        
        # Save results
        out_path = os.path.join(CONFIG["output_dir"], f"posterior_probs_coding_cost_{name}_beta{CONFIG['beta']}.json")
        with open(out_path, "w") as f:
            json.dump(post, f, indent=2, sort_keys=True)
        logger.info(f"Saved {len(post)} posterior entries to {out_path}")
        
        # Compute accuracy metrics
        compute_accuracy_metrics(post, classifier_likelihoods, name)
    
    total_time = time.time() - start_time
    logger.info(f"Pipeline completed in {total_time:.2f} seconds")

def compute_accuracy_metrics(posteriors, classifier_likelihoods, dataset_name):
    """Compute and display accuracy metrics"""
    correct_before = correct_after = improved = worsened = unchanged = total = 0
    
    for img_path in posteriors:
        if img_path not in classifier_likelihoods:
            continue
        
        gt = infer_ground_truth(img_path)
        if gt is None:
            continue
        
        before = classifier_likelihoods[img_path]
        after = posteriors[img_path]
        
        top_before = max(before, key=before.get)
        
        # Map posterior to coarse space
        coarse_post = defaultdict(float)
        for gen, val in after.items():
            label = GENERATOR_GROUPS[gen]
            coarse_post[label] += val
        
        top_after = max(coarse_post, key=coarse_post.get)
        
        if top_before == gt:
            correct_before += 1
        if top_after == gt:
            correct_after += 1
        
        if top_before != top_after:
            if top_after == gt:
                improved += 1
            elif top_before == gt:
                worsened += 1
        else:
            unchanged += 1
        
        total += 1
    
    if total > 0:
        logger.info(f"Accuracy Results for {dataset_name.upper()}:")
        logger.info(f"  Total images: {total}")
        logger.info(f"  Before: {correct_before/total:.2%} | After: {correct_after/total:.2%}")
        logger.info(f"  Improved: {improved} | Worsened: {worsened} | Unchanged: {unchanged}")

if __name__ == "__main__":
    main()
