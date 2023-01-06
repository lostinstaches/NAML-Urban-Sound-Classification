import os
import numpy as np
from sklearn.model_selection import train_test_split

from model import build_model

random_seed = 42

print("Loading data...", end=" ", flush=True)
features_path = os.path.abspath("data/processed/features_x.npy")
labels_path = os.path.abspath("data/processed/labels_y.npy")
X = np.load(features_path)
y = np.load(labels_path)
print("Done ✅!")

print("Splitting the data...", end=" ", flush=True)
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=random_seed
)
print("Done ✅!")

print("Building the model️...", end=" ", flush=True)
model = build_model()
print("Done ✅!")

print("Training the model...")
history = model.fit(
    x_train, y_train, validation_data=(x_val, y_val), batch_size=100, epochs=50
)
print("Done ✅!")

print("Saving the model...", end=" ", flush=True)
save_path = os.path.abspath("models/cnn_model_with_augmentation")
model.save(save_path)
print("Done ✅!")

# Without augmentation the accuracy is 0.708571
