import os
import tensorflow as tf
import numpy as np
import seaborn as sn
import pandas as pd
from librosa.feature import melspectrogram
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model():
    """ Build the CNN Model according to the J. Salamon and J.P. Bello paper """

    model = Sequential()
    model.add(SpecAugment(
        freq_mask_param=128,
        time_mask_param=128,
        n_freq_mask=1,
        n_time_mask=2,
        mask_value=-1
    ))
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_seed)

# 3. Data augmentation
print("Phase 3: Augment the data")

# TODO: Add the data augmentation part here

# 4. Build model
print("Phase 4: Building the model️")

model = build_model()

# 5. Fit model
print("Phase 5: Fit the model")

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=100, epochs=200)


# 6. Save model
print("Phase 6: Save the model")

MODEL_PATH = os.path.abspath('models/cnn_model_with_augmentation')
model.save(MODEL_PATH)

# 7. Predicting
print("Phase 7: Predicting")

prediction = model.predict(X_test)

# 8. Evaluating
print("Phase 8: Evaluating")

labels_predicted = np.argmax(model.predict(X_test, verbose=0), 1)
labels_true = np.argmax(y_test, 1)

accuracy = model.evaluate(X_test, y_test, batch_size=32)[1]
print("\nAccuracy = " + str(accuracy))

# Without augmentation the accuracy is 0.708571
