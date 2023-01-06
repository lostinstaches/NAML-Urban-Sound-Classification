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
from keras.optimizers.legacy.sgd import SGD


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
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer=SGD()
    )

    return model
