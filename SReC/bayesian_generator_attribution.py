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
    return zed_data

# Phase 2 & 3: Compute likelihood p(x|y) and prior p(y)
def compute_posterior(aeroblade, zed):
    results = []
    image_set = set(zed.keys())

    print(f"Processing {len(image_set)} images...")
    for image in image_set:
        print("image:", image)
        zed_scores = zed[image]
        labels = list(zed_scores.keys())
        scores = np.array([np.mean(zed_scores[label]) for label in labels])
        likelihood = softmax(scores)  # p(x|y)

        priors = []
        for label in labels:
            distance = aeroblade.get(label, {}).get(image, -1.0)
            prior = np.exp(-abs(distance))
            priors.append(prior)
        priors = np.array(priors)
        priors = priors / priors.sum()  # Normalize

        posterior = likelihood * priors
        posterior /= posterior.sum()  # Normalize

        results.append({
            'image': image,
            'labels': labels,
            'likelihood': likelihood.tolist(),
            'prior': priors.tolist(),
            'posterior': posterior.tolist(),
            'prediction': labels[np.argmax(posterior)]
        })
    return results

# Phase 4: Save predictions
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
