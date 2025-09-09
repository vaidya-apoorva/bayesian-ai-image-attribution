import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Load both JSON files
with open("data_coco_trained.json", "r") as f:
    data_coco = json.load(f)
with open("data_openimage.json", "r") as f:
    data_openimage = json.load(f)

# Define real and generated sources
real_sources = ['raise', 'coco']
generated_sources = ['dalle3', 'sdxl', 'midjourneyv5', 'dalle2']

# Helper to safely get paired gap values from both models
def get_paired_features(source, label):
    if source in data_coco and source in data_openimage:
        coco_vals = data_coco[source]
        openimage_vals = data_openimage[source]
        min_len = min(len(coco_vals), len(openimage_vals))
        if min_len == 0:
            print(f"Warning: No data found for source '{source}'")
        features = [[coco_vals[i], openimage_vals[i]] for i in range(min_len)]
        labels = [label] * min_len
        return np.array(features), np.array(labels)
    else:
        print(f"Warning: Missing source '{source}' in either 'data_coco' or 'data_openimage'")
    return np.empty((0, 2)), np.empty((0,))

# Build combined dataset
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

# Shuffle and split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check data and labels before splitting
print("Unique labels in the combined dataset:", np.unique(y))
print("Real sources data size:", len(X_all[:len(real_sources)]))
print("Generated sources data size:", len(X_all[len(real_sources):]))

# Shuffle and split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# After splitting, check for class distribution
print("Unique labels in training:", np.unique(y_train))
print("Unique labels in test:", np.unique(y_test))

# Continue with scaling and model training as before...


# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train model
model = Sequential([
    InputLayer(input_shape=(2,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

y_train = y_train.astype(int)
y_test = y_test.astype(int)
print("Unique labels in training:", np.unique(y_train))

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
print("Computed class weights:", class_weights)

class_weights_dict = {int(k): v for k, v in enumerate(class_weights)}
print("Final class_weights_dict keys:", class_weights_dict.keys())

model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop],
    class_weight=class_weights_dict
)


# model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate
# test_loss = model.evaluate(X_test_scaled, y_test)
# print(f"\nTest Mean Squared Error: {test_loss:.4f}")
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Predict function using ensemble features
def predict_realness(coco_gap, openimage_gap):
    features = np.array([[coco_gap, openimage_gap]])
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0, 0]

# Example usage
example_score = predict_realness(0.05, 0.01)
print(f"Predicted score (0 = real, 1 = generated): {example_score:.3f}")


# Visualization of decision boundary
plt.figure(figsize=(10, 6))

# Convert labels to integers for indexing
y_train_int = y_train.astype(int)

# Separate points based on their labels
real_points = X_train_scaled[y_train_int == 0]
generated_points = X_train_scaled[y_train_int == 1]


# Plot points for real (label 0) and generated (label 1)
plt.scatter(real_points[:, 0], real_points[:, 1], color='blue', label='Real', alpha=0.6)
plt.scatter(generated_points[:, 0], generated_points[:, 1], color='red', label='Generated', alpha=0.6)

# Create a grid of points to evaluate the model
h = 0.02  # step size for meshgrid
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the class for each point in the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)

# Labels and title
plt.title("Decision Boundary for Gap Features")
plt.xlabel("Coco Gap")
plt.ylabel("OpenImage Gap")
plt.legend()

plt.savefig('/mnt/ssd-data/vaidya/SReC/ensemble_regressor.png', dpi=300)
plt.show()