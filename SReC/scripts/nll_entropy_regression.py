import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to compute the "coding cost gap" feature (nll - entropy)
def compute_gap(nll, entropy):
    return nll - entropy

# Simulated dataset with 1000 samples.
# Each sample contains 6 values:
# [nll_coco, entropy_coco, nll_imagenet, entropy_imagenet, nll_raise, entropy_raise]
np.random.seed(42)
N = 1000
X_raw = np.random.rand(N, 6)  # Replace with your actual NLL & entropy values from SReC

# Compute the coding cost gap for each model.
gap_coco = compute_gap(X_raw[:, 0], X_raw[:, 1]) # Shape: (N,) and computes the gap for COCO using data from columns 0 and 1
gap_imagenet = compute_gap(X_raw[:, 2], X_raw[:, 3]) # Shape: (N,) and computes the gap for ImageNet using data from columns 2 and 3
gap_raise = compute_gap(X_raw[:, 4], X_raw[:, 5]) # Shape: (N,) and computes the gap for RAISE using data from columns 4 and 5
X_features = np.vstack([gap_coco, gap_imagenet, gap_raise]).T  # Shape: (N, 3)

# Simulated target values: 0 for real and 1 for generated.
# TODO: Replace this with your actual labels.
y = np.random.randint(0, 2, N)  # 0 (real) or 1 (generated)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")  
print(f"Example feature vector: {X_train[0]}")
print(f"Example target value: {y_train[0]}")

# Scale the features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(f"X train scaled: {X_train_scaled[0]}")
X_test_scaled = scaler.transform(X_test)
print(f"X test scaled: {X_test_scaled[0]}")

# Define a simple neural network regressor.
# The output layer uses a sigmoid activation to constrain outputs between 0 and 1.
model = Sequential([
    InputLayer(input_shape=(3,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model.
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
# 31 steps (batches) per epoch, 50 epochs, 0.1 validation split, verbose=1

# Evaluate the model on the test set.
test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Mean Squared Error: {test_loss:.4f}")

# Define a function to predict the score for a new image.
# A score closer to 0 indicates a real image; closer to 1 indicates a generated image.
def predict_realness(nll_coco, entropy_coco, nll_imagenet, entropy_imagenet, nll_raise, entropy_raise):
    gap_coco = compute_gap(nll_coco, entropy_coco)
    gap_imagenet = compute_gap(nll_imagenet, entropy_imagenet)
    gap_raise = compute_gap(nll_raise, entropy_raise)
    features = np.array([[gap_coco, gap_imagenet, gap_raise]])
    features_scaled = scaler.transform(features)
    score = model.predict(features_scaled)[0,0]
    return score

# Example prediction:
example_score = predict_realness(0.5, 0.4, 0.6, 0.55, 0.7, 0.65)
print(f"Predicted score (0=real, 1=generated): {example_score:.3f}")