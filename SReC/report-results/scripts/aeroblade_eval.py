import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def infer_label(dataset_name):
    return 0 if dataset_name in ['coco', 'raise'] else 1

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, help='Folder with *_distances.csv files')
parser.add_argument('--output', help='Path to save metrics as JSON')
parser.add_argument('--real_percentile', type=float, default=40.0, help='Target TPR percentile threshold for real images')
args = parser.parse_args()

y_true, distances, dataset_names = [], [], []
real_distances = []

# Collect data
for fname in os.listdir(args.input_dir):
    if fname.endswith('_distances.csv'):
        dataset = fname.replace('_distances.csv', '')
        label = infer_label(dataset)

        df = pd.read_csv(os.path.join(args.input_dir, fname))
        for dist in df['distance']:
            y_true.append(label)
            distances.append(dist)
            dataset_names.append(dataset)
            if label == 0:
                real_distances.append(dist)

# Compute threshold based on real images
threshold = np.percentile(real_distances, args.real_percentile)

# Classify: 1 = fake if LPIPS > threshold
y_pred = [1 if d > threshold else 0 for d in distances]

# Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Threshold (95th percentile of real): {threshold:.6f}")
print(f"AEROBLADE Accuracy: {acc:.4f}")
print(f"AEROBLADE F1 Score: {f1:.4f}")

# Per-dataset accuracy
dataset_acc = {}
for ds in np.unique(dataset_names):
    idxs = [i for i, d in enumerate(dataset_names) if d == ds]
    acc_ds = accuracy_score([y_true[i] for i in idxs], [y_pred[i] for i in idxs])
    dataset_acc[ds] = round(acc_ds, 4)
    print(f"{ds}: Accuracy = {acc_ds:.4f}")

# Save results
if args.output:
    import json
    results = {
        'threshold': round(threshold, 6),
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4),
        'per_dataset_accuracy': dataset_acc
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {args.output}")
