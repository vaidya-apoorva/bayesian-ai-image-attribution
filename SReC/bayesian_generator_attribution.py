import os
import json
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from scipy.special import softmax

# Phase 1: Load AeroBlade and ZED outputs
def load_aeroblade_outputs(directory):
    aeroblade_data = {}
    for file_path in glob(os.path.join(directory, '*_distances.csv')):
        model_name = Path(file_path).stem.replace('_distances', '')
        df = pd.read_csv(file_path)
        df['file'] = df['file'].apply(lambda x: Path(x).name.replace('.jpg.png', '.jpg'))
        aeroblade_data[model_name] = dict(zip(df['file'], df['distance']))
    return aeroblade_data

def load_zed_outputs(*json_files):
    zed_data = {}
    for file_path in json_files:
        with open(file_path) as f:
            data = json.load(f)
        for image, preds in data.items():
            zed_data[image] = preds

    print("\n--- ZED DATA SAMPLE ---")
    for i, (k, v) in enumerate(zed_data.items()):
        print(f"Image: {k}")
        print("Type:", type(v))
        if isinstance(v, dict):
            print("Keys:", list(v.keys())[:5])
        elif isinstance(v, list):
            print("List length:", len(v))
            if len(v) > 0:
                print("First element type:", type(v[0]))
                print("First element:", v[0])
        else:
            print("Value:", v)
        print()
        if i >= 2:
            break  # only show 3 examples
    print("------------------------\n")
    return zed_data

# Phase 2 & 3: Compute likelihood p(x|y) and prior p(y)
# def compute_posterior(aeroblade, zed):
#     results = []
#     image_set = set(zed.keys())
#
#     print(f"Processing {len(image_set)} images...")
#     for image in image_set:
#         print("image:", image)
#         zed_scores = zed[image]
#         labels = list(zed_scores.keys())
#         scores = np.array([np.mean(zed_scores[label]) for label in labels])
#         likelihood = softmax(scores)  # p(x|y)
#
#         priors = []
#         for label in labels:
#             distance = aeroblade.get(label, {}).get(image, -1.0)
#             prior = np.exp(-abs(distance))
#             priors.append(prior)
#         priors = np.array(priors)
#         priors = priors / priors.sum()  # Normalize
#
#         posterior = likelihood * priors
#         posterior /= posterior.sum()  # Normalize
#
#         results.append({
#             'image': image,
#             'labels': labels,
#             'likelihood': likelihood.tolist(),
#             'prior': priors.tolist(),
#             'posterior': posterior.tolist(),
#             'prediction': labels[np.argmax(posterior)]
#         })
#     return results

# Phase 4: Save predictions

def compute_posterior(aeroblade, zed):
    # ZED format is: {label: [score_for_image_0, score_for_image_1, ...]}
    labels = list(zed.keys())
    if not labels:
        print("⚠️ No labels found in ZED data.")
        return []

    # Build a sorted list of images from AeroBlade outputs (union across models)
    all_images = set()
    for model_dict in aeroblade.values():
        all_images.update(model_dict.keys())
    images = sorted(all_images)

    if not images:
        print("⚠️ No images found in AeroBlade outputs.")
        return []

    # Determine usable length = min(len(images), min(len(zed[label]) for each label))
    zed_lengths = {label: len(zed[label]) for label in labels}
    usable_len = min(len(images), *(zed_lengths.values()))
    if usable_len < len(images) or any(zed_lengths[l] != usable_len for l in labels):
        print(f"⚠️ Length mismatch: using first {usable_len} items.")
        print(f"   Images available: {len(images)}; ZED lengths: {zed_lengths}")

    results = []
    print(f"Processing {usable_len} images across {len(labels)} labels...")

    for i in range(usable_len):
        image = images[i]

        # Collect ZED scores for this image index across labels
        row_scores = []
        for label in labels:
            try:
                val = zed[label][i]
            except Exception:
                val = 0.0
            # Guard against NaNs
            if val is None or not np.isfinite(val):
                val = 0.0
            row_scores.append(float(val))

        scores = np.asarray(row_scores, dtype=np.float64)
        likelihood = softmax(scores)  # p(x|y)

        # Priors from AeroBlade distances for this image per label
        priors = []
        for label in labels:
            distance = aeroblade.get(label, {}).get(image, -1.0)
            prior = np.exp(-abs(distance))
            priors.append(prior)
        priors = np.asarray(priors, dtype=np.float64)

        pri_sum = priors.sum()
        if pri_sum <= 0 or not np.isfinite(pri_sum):
            priors = np.ones_like(priors) / len(priors)
        else:
            priors /= pri_sum

        posterior = likelihood * priors
        post_sum = posterior.sum()
        if post_sum > 0 and np.isfinite(post_sum):
            posterior /= post_sum
        else:
            posterior = np.ones_like(posterior) / len(posterior)

        results.append({
            'image': image,
            'labels': labels,
            'likelihood': likelihood.tolist(),
            'prior': priors.tolist(),
            'posterior': posterior.tolist(),
            'prediction': labels[int(np.argmax(posterior))]
        })

    return results

def save_results(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


# Main function
def run_pipeline():
    aeroblade_dir = '/mnt/ssd-data/vaidya/aeroblade/aeroblade_output'
    zed_files = [
        '/mnt/ssd-data/vaidya/SReC/data_coco_trained.json',
        '/mnt/ssd-data/vaidya/SReC/data_openimage.json'
    ]
    output_path = '/mnt/ssd-data/vaidya/bayesian_predictions.json'

    print("Loading AeroBlade outputs...")
    aeroblade = load_aeroblade_outputs(aeroblade_dir)

    print("Loading ZED outputs...")
    zed = load_zed_outputs(*zed_files)

    print("Computing posterior probabilities...")
    results = compute_posterior(aeroblade, zed)

    print(f"Saving results to {output_path}")
    save_results(results, output_path)

if __name__ == '__main__':
    run_pipeline()
