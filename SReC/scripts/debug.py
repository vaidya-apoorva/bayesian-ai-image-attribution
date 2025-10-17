import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load both JSON files
with open("data_coco_trained.json", "r") as f:
    data_coco = json.load(f)
with open("data_openimage.json", "r") as f:
    data_openimage = json.load(f)

# Define real and generated sources
real_sources = ['raise', 'coco']
generated_sources = ['dalle3', 'sdxl', 'midjourneyv5', 'dalle2']

# Helper to get paired features
def get_paired_features(source, label):
    if source in data_coco and source in data_openimage:
        coco_vals = data_coco[source]
        openimage_vals = data_openimage[source]
        min_len = min(len(coco_vals), len(openimage_vals))
        features = [[coco_vals[i], openimage_vals[i]] for i in range(min_len)]
        labels = [label] * min_len
        return np.array(features), np.array(labels)
    return np.empty((0, 2)), np.empty((0,))

# Build dataset
X_all, y_all = [], []
for source in real_sources:
    X_r, y_r = get_paired_features(source, 0)
    X_all.append(X_r)
    y_all.append(y_r)
for source in generated_sources:
    X_f, y_f = get_paired_features(source, 1)
    X_all.append(X_f)
    y_all.append(y_f)

X = np.vstack(X_all)
y = np.concatenate(y_all)

# Shuffle + split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = Sequential([
    InputLayer(input_shape=(2,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Predict function
def predict_realness(coco_gap, openimage_gap, threshold=0.5):
    features = np.array([[coco_gap, openimage_gap]])
    features_scaled = scaler.transform(features)
    pred_score = model.predict(features_scaled)[0, 0]
    label = 1 if pred_score >= threshold else 0
    return pred_score, label

# Example prediction
score, label = predict_realness(0, 0)
print(f"Predicted Score: {score:.3f}, Predicted Label: {'Generated' if label == 1 else 'Real'}")

# === Visualization 1: Decision Boundary on Train Set ===
plt.figure(figsize=(10, 6))
y_train_int = y_train.astype(int)
real_points = X_train_scaled[y_train_int == 0]
generated_points = X_train_scaled[y_train_int == 1]
plt.scatter(real_points[:, 0], real_points[:, 1], color='blue', label='Real (Train)', alpha=0.6)
plt.scatter(generated_points[:, 0], generated_points[:, 1], color='red', label='Generated (Train)', alpha=0.6)
h = 0.02
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.title("Decision Boundary on Training Set")
plt.xlabel("COCO Gap")
plt.ylabel("OpenImage Gap")
plt.legend()
plt.savefig('/mnt/ssd-data/vaidya/SReC/ensemble_regressor_debug.png', dpi=300)
plt.show()

# === Visualization 2: Predicted Labels on Test Data ===
y_pred_probs = model.predict(X_test_scaled)
y_pred_labels = (y_pred_probs >= 0.5).astype(int).flatten()
real_pred = X_test_scaled[y_pred_labels == 0]
fake_pred = X_test_scaled[y_pred_labels == 1]

plt.figure(figsize=(10, 6))
plt.scatter(real_pred[:, 0], real_pred[:, 1], color='green', label='Predicted Real', alpha=0.6)
plt.scatter(fake_pred[:, 0], fake_pred[:, 1], color='red', label='Predicted Generated', alpha=0.6)
plt.title("Predicted Labels on Test Data")
plt.xlabel("COCO Gap")
plt.ylabel("OpenImage Gap")
plt.legend()
plt.grid(True)
plt.savefig('/mnt/ssd-data/vaidya/SReC/test_predictions.png', dpi=300)
plt.show()

# === Report: Test Confusion and Classification ===
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, labels=[0, 1], target_names=["Real", "Generated"]))


# === Visualization 3: Predicted Scores on Real Validation Images ===
# Replace below with real validation gaps
val_data = [(0.3, 0.2), (0.25, 0.1), (0.35, 0.3)]  # replace with real values
val_scores = [predict_realness(c, o)[0] for c, o in val_data]

plt.figure(figsize=(8, 5))
plt.hist(val_scores, bins=20, color='purple', alpha=0.7)
plt.axvline(0.5, color='black', linestyle='--', label='Threshold')
plt.title("Predicted Scores for Real Validation Images")
plt.xlabel("Predicted Score (0 = Real, 1 = Generated)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('/mnt/ssd-data/vaidya/SReC/validation_real_scores.png', dpi=300)
plt.show()
