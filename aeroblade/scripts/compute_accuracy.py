import json
import os

# Load classifier likelihoods and posterior results
with open("/mnt/ssd-data/vaidya/SReC/results/classifier_results.json") as f:
    classifier_likelihoods = json.load(f)

with open("/mnt/ssd-data/vaidya/aeroblade/results/posterior_probs.json") as f:
    posteriors = json.load(f)

# Mapping folder name to class label
folder_to_class = {
    'coco': 'Real',
    'real': 'Real',
    'raise': 'Real',
    'dalle2': 'DALL-E',
    'dalle3': 'DALL-E',
    'sdxl': 'StableDiffusion',
    'midjourneyv5': 'MidJourney'
}

def get_true_label(img_path):
    folder = os.path.basename(os.path.dirname(img_path)).lower()
    return folder_to_class.get(folder)

def compute_accuracy(predictions, true_labels):
    correct = 0
    total = 0
    for img_path, probs in predictions.items():
        true = true_labels.get(img_path)
        if not true:
            continue
        pred = max(probs, key=probs.get)
        total += 1
        if pred == true:
            correct += 1
    return correct, total, correct / total if total > 0 else 0.0

# Build true label dictionary from either source
all_paths = set(classifier_likelihoods.keys()).union(posteriors.keys())
true_labels = {p: get_true_label(p) for p in all_paths if get_true_label(p) is not None}

# Compute accuracies
correct_cls, total_cls, acc_cls = compute_accuracy(classifier_likelihoods, true_labels)
correct_post, total_post, acc_post = compute_accuracy(posteriors, true_labels)

# Print comparison
print(f"Classifier-only Accuracy: {acc_cls:.4f} ({correct_cls}/{total_cls})")
print(f"Bayesian Posterior Accuracy: {acc_post:.4f} ({correct_post}/{total_post})")
