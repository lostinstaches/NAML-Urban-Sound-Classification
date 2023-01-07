import os

from keras.preprocessing.image import ImageDataGenerator
from model import build_model
from utils import load_data, plot_results

print("Prepare data generators...", end=" ", flush=True)
validation_fold = 1
dataset_path = "data/UrbanSound8K/augmented_data_images"
train_data, train_labels, validation_data, validation_labels = load_data(
    dataset_path=dataset_path,
    validation_fold=validation_fold,
    modifications_to_include=["original"],
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

plot_results(history)

print("Saving the model...")
save_path = os.path.abspath(
    f"models/cnn_model_without_augmentation_fold{validation_fold}"
)
model.save(save_path)
print("Done ✅!")

# Without augmentation the accuracy is 0.708571
