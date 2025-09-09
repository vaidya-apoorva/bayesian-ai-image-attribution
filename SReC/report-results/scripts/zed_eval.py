import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Helper: infer label from filename
def infer_label_from_filename(fname):
    if any(real in fname for real in ['coco', 'raise']):
        return 0  # Real
    else:
        return 1  # Synthetic

# Helper: infer dataset name from filename (e.g., 'coco_d0.json' → 'coco')
def infer_dataset_from_filename(fname):
    return fname.split('_d0')[0]

# Parse args
parser = argparse.ArgumentParser(description="Compute overall and per-dataset metrics from JSON files")
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing *_d0.json files')
parser.add_argument('--output', type=str, required=True, help='Path to save accuracy and F1 score')
args = parser.parse_args()

# Collect all data
X, y, dataset_names = [], [], []

for filename in os.listdir(args.input_dir):
    if filename.endswith('_d0.json'):
        full_path = os.path.join(args.input_dir, filename)
        label = infer_label_from_filename(filename)
        dataset = infer_dataset_from_filename(filename)

        with open(full_path, 'r') as f:
            data = json.load(f)
            for gap in data.values():
                X.append([gap])
                y.append(label)
                dataset_names.append(dataset)

X = np.array(X)
y = np.array(y)
dataset_names = np.array(dataset_names)

# Train/test split with stratification on labels, keep dataset_names aligned
X_train, X_test, y_train, y_test, dataset_train, dataset_test = train_test_split(
    X, y, dataset_names, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# Compute overall metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Overall Accuracy: {acc:.4f}")
print(f"Overall F1 Score: {f1:.4f}\n")

# Compute per-dataset accuracy
unique_datasets = np.unique(dataset_test)
per_dataset_acc = {}
print("Per-dataset accuracy:")
for d in unique_datasets:
    idx = dataset_test == d
    acc_d = accuracy_score(y_test[idx], y_pred[idx])
    per_dataset_acc[d] = round(acc_d, 4)
    print(f"{d}: Accuracy = {acc_d:.4f}")

# Save results
results = {
    'overall': {
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4)
    },
    'per_dataset_accuracy': per_dataset_acc
}

with open(args.output, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nSaved results to {args.output}")
