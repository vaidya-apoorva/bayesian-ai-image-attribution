import os
import joblib
import json
import numpy as np

# 1. Load KDE models from directory
kde_model_dir = "/mnt/ssd-data/vaidya/aeroblade/models/generator_kdes"
kde_models = {}
for fname in os.listdir(kde_model_dir):
    if fname.endswith(".joblib") and fname.startswith("kde_model_"):
        gen_name = fname.replace("kde_model_", "").replace(".joblib", "")
        model_path = os.path.join(kde_model_dir, fname)
        kde_models[gen_name] = joblib.load(model_path)
print(f"Loaded KDE models for: {list(kde_models.keys())}")

# 2. Compute prior from KDE
def compute_prior(distance, kde_model, epsilon=1e-10):
    if kde_model is None:
        return epsilon
    score = kde_model.score_samples([[distance]])[0]  # log density
    return max(np.exp(score), epsilon)

# 3. Compute posterior combining likelihood and KDE prior
def compute_posteriors(classifier_likelihoods, distances, kde_models):
    posteriors = {}

    for img_path, likelihoods in classifier_likelihoods.items():
        img_name = os.path.basename(img_path)  # get only filename, e.g. "img_A_fish_eye_lens_shows_the_corner_...png"
        d = distances.get(img_name)
        if d is None:
            print(f"[WARN] Missing distance for {img_path}, skipping.")
            continue

        if isinstance(d, dict) and 'distance' in d:
            d = d['distance']

        posterior_unnorm = {}
        for gen, likelihood in likelihoods.items():
            kde = kde_models.get(gen)
            prior = compute_prior(d, kde)
            posterior_unnorm[gen] = likelihood * prior

        s = sum(posterior_unnorm.values())
        if s == 0:
            print(f"[WARN] Zero posterior sum for {img_path}, adjusting.")
            s = 1e-8

        posterior_norm = {gen: val / s for gen, val in posterior_unnorm.items()}
        posteriors[img_path] = posterior_norm

    return posteriors

# 4. Load your JSON data from given paths
image_distances_path = "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/image_distances.json"
classifier_likelihoods_path = "/mnt/ssd-data/vaidya/SReC/results/classifier_results.json"

with open(classifier_likelihoods_path) as f:
    classifier_likelihoods = json.load(f)

with open(image_distances_path) as f:
    image_distances = json.load(f)

# 5. Compute posteriors and save
posteriors = compute_posteriors(classifier_likelihoods, image_distances, kde_models)

output_path = "/mnt/ssd-data/vaidya/aeroblade/results/posterior_probs.json"

with open(output_path, "w") as f:
    json.dump(posteriors, f, indent=2, sort_keys=True)

print(f"Posterior probabilities saved to {output_path}")

