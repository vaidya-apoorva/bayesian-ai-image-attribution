import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate ZED model accuracy using gap values.")
parser.add_argument('--input', type=str, required=True, help='Path to input JSON file with gap values')
parser.add_argument('--output', type=str, default='zed_metrics.json', help='Path to output JSON file for results')
args = parser.parse_args()

# Load gap values
with open(args.input, 'r') as f:
    data = json.load(f)

# Prepare features, labels, and dataset names
X = []
y = []
dataset_names = []

# Real images: coco and raise (label 0)
for v in data.get('coco', []):
    X.append([v])
    y.append(0)
    dataset_names.append('coco')

for v in data.get('raise', []):
    X.append([v])
    y.append(0)
    dataset_names.append('raise')

# Synthetic images: dalle3, sdxl, midjourneyv5
for key in ['dalle3', 'sdxl', 'midjourneyv5']:
    for v in data.get(key, []):
        X.append([v])
        y.append(1)
        dataset_names.append(key)

X = np.array(X)
y = np.array(y)
dataset_names = np.array(dataset_names)

# Train/test split - stratify by label
X_train, X_test, y_train, y_test, dataset_train, dataset_test = train_test_split(
    X, y, dataset_names, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)

# Overall metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Overall Accuracy: {acc:.4f}")
print(f"Overall F1 Score: {f1:.4f}\n")

# Per-dataset metrics
unique_datasets = np.unique(dataset_test)
print("Per-dataset metrics:")
per_dataset_acc = {}
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
