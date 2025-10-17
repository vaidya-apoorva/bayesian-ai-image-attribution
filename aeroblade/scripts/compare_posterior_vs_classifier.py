import json
import os
import matplotlib.pyplot as plt

# File paths
classifier_path = "/mnt/ssd-data/vaidya/SReC/results/classifier_results.json"
posterior_path = "/mnt/ssd-data/vaidya/aeroblade/results/posterior_probs.json"
save_dir = "/mnt/ssd-data/vaidya/aeroblade/results/delta_top10"
os.makedirs(save_dir, exist_ok=True)

# Load predictions
with open(classifier_path) as f:
    classifier_likelihoods = json.load(f)
with open(posterior_path) as f:
    posteriors = json.load(f)

# Compute delta confidence for predicted class
delta_scores = []
for img_path, cls_probs in classifier_likelihoods.items():
    if img_path not in posteriors:
        continue
    pred_class = max(cls_probs, key=cls_probs.get)
    cls_conf = cls_probs[pred_class]
    post_conf = posteriors[img_path][pred_class]
    delta = post_conf - cls_conf
    delta_scores.append((img_path, pred_class, delta, cls_conf, post_conf))

# Sort by improvement in confidence
top_10 = sorted(delta_scores, key=lambda x: x[2], reverse=True)[:10]

# Plot and save each
for img_path, pred_class, delta, cls_val, post_val in top_10:
    plt.figure(figsize=(5, 4))
    plt.bar(['Classifier', 'Posterior'], [cls_val, post_val], color=['orange', 'green'])
    plt.ylim(0, 1)
    plt.title(f"{os.path.basename(img_path)}\nClass: {pred_class}, Δ = {delta:.4f}")
    plt.ylabel("Confidence")
    plt.tight_layout()

    img_name = os.path.basename(img_path).replace(".jpg.png", "").replace(".png", "")
    save_path = os.path.join(save_dir, f"delta_conf_{img_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")
