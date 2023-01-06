import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from model import build_model

print("Prepare data generators...", end=" ", flush=True)
train_folds = [2, 3]
validation_fold = 1


train_data = np.load(
    f"data/UrbanSound8K/augmented_data_images/train/fold{train_folds[0]}/images.npy"
)
train_labels = np.load(
    f"data/UrbanSound8K/augmented_data_images/train/fold{train_folds[0]}/labels.npy"
)
for train_fold in train_folds[1:]:
    train_data = np.concatenate(
        (
            train_data,
            np.load(
                f"data/UrbanSound8K/augmented_data_images/train/fold{train_fold}/images.npy"
            ),
        )
    )
    train_labels = np.concatenate(
        (
            train_labels,
            np.load(
                f"data/UrbanSound8K/augmented_data_images/train/fold{train_fold}/labels.npy"
            ),
        )
    )
train_data = train_data.reshape((train_data.shape[0], 128, 130, 1))


validation_data = np.load(
    f"data/UrbanSound8K/augmented_data_images/validation/fold{validation_fold}/images.npy"
)
validation_data = validation_data.reshape((validation_data.shape[0], 128, 130, 1))

validation_labels = np.load(
    f"data/UrbanSound8K/augmented_data_images/validation/fold{validation_fold}/labels.npy"
)
image_generator = ImageDataGenerator()

print("Done ✅!")


print("Building the model️...", end=" ", flush=True)
model = build_model()
print("Done ✅!")


print("Training the model...")
history = model.fit(
    image_generator.flow(train_data, train_labels, batch_size=128),
    validation_data=image_generator.flow(
        validation_data, validation_labels, batch_size=128
    ),
    epochs=50,
)
print("Done ✅!")


print("Saving the model...")
save_path = os.path.abspath(f"models/cnn_model_with_augmentation_fold{validation_fold}")
model.save(save_path)
print("Done ✅!")

# Without augmentation the accuracy is 0.708571
