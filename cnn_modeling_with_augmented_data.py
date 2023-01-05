import os
from keras.models import Sequential
from keras.layers import (
    Convolution2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Activation,
    Flatten,
)
from keras.regularizers import l2

from jams_to_image_data_generator import JamsToImageDataGenerator


def build_model():
    """Build the CNN Model according to the J. Salamon and J.P. Bello paper"""

    model = Sequential()
    model.add(
        Convolution2D(
            24,
            (5, 5),
            strides=(1, 1),
            padding="valid",  # no padding and it assumes that all the dimensions are valid
            input_shape=(128, 128, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation("relu"))

    model.add(Convolution2D(48, (5, 5), strides=(1, 1), padding="valid"))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation("relu"))

    model.add(Convolution2D(48, (5, 5), strides=(1, 1), padding="valid"))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=l2(0.001)))  # l2 — penalty factor
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="SGD"
    )
    return model


random_seed = 42

# 1. Load from the processed data
print("Phase 1: Loading data")

train_fold_path = os.path.abspath("data/UrbanSound8K/augmented_data/train/fold1")
validation_fold_path = os.path.abspath(
    "data/UrbanSound8K/augmented_data/validation/fold1"
)
train_data_generator = JamsToImageDataGenerator(
    fold_path=train_fold_path, mini_batch_size=128
)
validation_data_generator = JamsToImageDataGenerator(
    fold_path=validation_fold_path, mini_batch_size=128
)

# 3. Build model
print("Phase 3: Building the model️")

model = build_model()

# 4. Fit model
print("Phase 4: Fit the model")

history = model.fit_generator(
    train_data_generator, validation_data=validation_data_generator, epochs=50
)


# 5. Save model
print("Phase 5: Save the model")

MODEL_PATH = os.path.abspath("models/cnn_model_with_augmentation")
model.save(MODEL_PATH)


# Without augmentation the accuracy is 0.708571
