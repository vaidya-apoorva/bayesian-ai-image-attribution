import os
import json
import joblib
import numpy as np
from collections import defaultdict

# KDE directories
kde_dirs = {
    "coco": "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED/coco_trained_JPEG_80/joblib",
    "openimages": "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED/openimages_trained_JPEG_80/joblib",
}

# File paths
classifier_likelihoods_path = "/mnt/ssd-data/vaidya/SReC/results/classifier_result_updated.json"
image_gaps_path = "/mnt/ssd-data/vaidya/SReC/results/image_coding_cost_open_images.json"
output_dir = "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED_classifier_results/JPEG80"
beta = 3.0  # Prior influence factor

# Generator → Coarse label mapping
GENERATOR_GROUPS = {
    "coco_d0": "Real",
    "raise_d0": "Real",
    "dalle2_d0": "DALL-E",
    "dalle3_d0": "DALL-E",
    "midjourneyV5_d0": "MidJourney",
    "sdxl_d0": "StableDiffusion",
}

# Ground truth inference from folder
def infer_ground_truth(path):
    if "coco" in path or "raise" in path:
        return "Real"
    elif "dalle2" in path or "dalle3" in path:
        return "DALL-E"
    elif "midjourney" in path.lower():
        return "MidJourney"
    elif "sdxl" in path.lower():
        return "StableDiffusion"
    else:
        return None

# Load data
with open(classifier_likelihoods_path) as f:
    classifier_likelihoods = json.load(f)
with open(image_gaps_path) as f:
    image_gaps = json.load(f)

def compute_prior(gap, kde_model, epsilon=1e-10):
    if kde_model is None:
        return epsilon
    log_density = kde_model.score_samples([[gap]])[0]
    return max(np.exp(log_density), epsilon)

def compute_posteriors(likelihoods, gaps, kde_models, beta=3.0):
    posteriors = {}
    for img_path, like_dict in likelihoods.items():
        img_name = os.path.basename(img_path)
        entry = gaps.get(img_name)
        if entry is None:
            print(f"[WARN] Missing gap for {img_name}, skipping.")
            continue
        gap = entry if isinstance(entry, (int, float)) else entry.get("gap")
        if gap is None:
            print(f"[WARN] No numeric gap for {img_name}, skipping.")
            continue

        log_unnorm = {}
        for gen in kde_models:
            coarse_label = GENERATOR_GROUPS.get(gen)
            if coarse_label is None:
                continue
            like = max(like_dict.get(coarse_label, 0.0), 1e-10)
            prior = compute_prior(gap, kde_models.get(gen))
            log_unnorm[gen] = np.log(like) + beta * np.log(prior)

        max_log = max(log_unnorm.values())
        exp_unnorm = {gen: np.exp(val - max_log) for gen, val in log_unnorm.items()}
        total = sum(exp_unnorm.values()) or 1e-8
        posteriors[img_path] = {gen: val / total for gen, val in exp_unnorm.items()}
    return posteriors

# Main loop
for name, kde_dir in kde_dirs.items():
    kde_models = {}
    for fname in os.listdir(kde_dir):
        if fname.startswith("kde_model_") and fname.endswith(".joblib"):
            gen = fname[len("kde_model_"):-len(".joblib")]
            kde_models[gen] = joblib.load(os.path.join(kde_dir, fname))
    print(f"[INFO] Loaded KDEs for {name}: {sorted(kde_models)}")

    post = compute_posteriors(classifier_likelihoods, image_gaps, kde_models, beta=beta)

    # Debug output
    print("\n[DEBUG] Sample Prior Influence:")
    for img_path in list(post)[:5]:
        img_name = os.path.basename(img_path)
        gap = image_gaps.get(img_name)
        if isinstance(gap, dict):
            gap = gap.get("gap")
        print(f"\n[IMAGE] {img_name} | Gap: {gap}")
        for gen in sorted(kde_models):
            prior = compute_prior(gap, kde_models.get(gen))
            print(f"  {gen:<15} prior = {prior:.6f}")


    # Save raw posterior
    out_path = os.path.join(output_dir, f"posterior_probs_coding_cost_{name}_beta{beta}.json")
    with open(out_path, "w") as f:
        json.dump(post, f, indent=2, sort_keys=True)
    print(f"[INFO] Saved {len(post)} posterior entries to {out_path}")

    # Compute accuracy changes
    correct_before = 0
    correct_after = 0
    improved = 0
    worsened = 0
    unchanged = 0
    total = 0

    coarse_posteriors = {}

    for img_path in post:
        if img_path not in classifier_likelihoods:
            continue
        gt = infer_ground_truth(img_path)
        if gt is None:
            continue

        before = classifier_likelihoods[img_path]
        after = post[img_path]

        top_before = max(before, key=before.get)

        # Map posterior to coarse space
        coarse_post = defaultdict(float)
        for gen, val in after.items():
            label = GENERATOR_GROUPS[gen]
            coarse_post[label] += val
        coarse_post = dict(coarse_post)
        coarse_posteriors[img_path] = coarse_post

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

    print(f"\n[INFO] Accuracy Shift for {name.upper()}:")
    print(f"        Total images evaluated: {total}")
    print(f"        Accuracy Before: {correct_before / total:.2%}")
    print(f"        Accuracy After:  {correct_after / total:.2%}")
    print(f"        → Improved: {improved}")
    print(f"        → Worsened: {worsened}")
    print(f"        → Unchanged: {unchanged}")

    # Optional: save coarse-style posterior
    # coarse_out_path = os.path.join(output_dir, f"posterior_probs_{name}_beta{beta}_coarse.json")
    # with open(coarse_out_path, "w") as f:
    #     json.dump(coarse_posteriors, f, indent=2, sort_keys=True)
