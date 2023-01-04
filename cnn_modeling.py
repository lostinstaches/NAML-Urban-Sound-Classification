import os
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

def build_model():
    """ Build the CNN Model according to the J. Salamon and J.P. Bello paper """

    model = Sequential()
    model.add(
        Convolution2D(
            24,
            (5, 5),
            strides=(1, 1),
            padding='valid',  # no padding and it assumes that all the dimensions are valid
            input_shape=(128, 128, 1)
        )
    )
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, (5, 5), strides=(1, 1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, (5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=l2(0.001)))  # l2 — penalty factor
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="SGD")
    return model


random_seed = 42

# 1. Load from the processed data
print("Phase 1: Loading data")

FEATURES_PATH = os.path.abspath('data/processed/features_x.npy')
LABELS_PATH = os.path.abspath('data/processed/labels_y.npy')

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)
print(X.shape)

# 2. Split data — Train / Validation / Test
print("Phase 2: Splitting the data")

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=random_seed)

# 3. Build model
print("Phase 3: Building the model️")

model = build_model()

# 4. Fit model
print("Phase 4: Fit the model")

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=100, epochs=50)


# 5. Save model
print("Phase 5: Save the model")

MODEL_PATH = os.path.abspath('models/cnn_model_with_augmentation')
model.save(MODEL_PATH)


# Without augmentation the accuracy is 0.708571
