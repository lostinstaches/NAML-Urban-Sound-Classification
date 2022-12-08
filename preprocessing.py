import glob
import os
import librosa  # librosa is a python package for music and audio analysis
import numpy as np
import pandas as pd
from librosa.feature import melspectrogram


def log_mel_spec(audio_directory, fold_directories):
    """" Converting dataset into Log Mel Spectograms """

    labels = []
    log_mel_spectrogram = []
    extension = "*.wav"

    for fold_directory in fold_directories:
        for filename in glob.glob(os.path.join(audio_directory, fold_directory, extension)):
            # The second number in the file name is the label
            label = filename.split('fold')[1].split('-')[1]
            labels.append(label)
            # sr = audio rate
            time_series, sr = librosa.load(filename, sr=44100)

            input_patch_length = 3 * sr
            if len(time_series) >= input_patch_length:
                # Convert a power spectrogram (amplitude squared) to decibel (dB) units
                spectogram_db = librosa.power_to_db(melspectrogram(
                    time_series[:input_patch_length],
                    sr=sr,
                    n_fft=1034,  # length of the FFT window
                    hop_length=1034)
                )
            else:
                # If the time series is less than 3 seconds, repeat series until it is 3 secs long
                while len(time_series) < input_patch_length:
                    time_series = np.concatenate((time_series, time_series))

                spectogram_db = librosa.power_to_db(melspectrogram(
                    time_series[:input_patch_length],
                    sr=sr,
                    n_fft=1034,  # length of the FFT window
                    hop_length=1034)
                )

            log_mel_spectrogram.append(spectogram_db)
        return np.array(log_mel_spectrogram), np.array(labels, dtype=np.int32)


def encode(labels):
    """" One Hot Encoding of Labels """

    labels_total = len(labels)
    unique_labels_total = len(np.unique(labels))
    one_hot_encoded = np.zeros((labels_total, unique_labels_total))
    one_hot_encoded[np.arange(labels_total), labels] = 1
    return one_hot_encoded


def create_file(final_path, filename):
    """" Create files """

    new_path = os.path.join(os.getcwd(), final_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return os.path.join(final_path, filename)


AUDIO_PATH = os.path.abspath('data/UrbanSound8K/audio/')
METADATA_PATH = os.path.abspath('data/UrbanSound8K/metadata/UrbanSound8K.csv')

# Sneak peek into the metadata file
metadata_df = pd.read_csv(METADATA_PATH,
                          usecols=["slice_file_name", "fold", "classID"],
                          dtype={"fold": "uint8", "classID": "uint8"})

print(metadata_df)  # -> [8732 rows x 3 columns]

PROCESSED_PATH = 'data/processed'

feature_file = create_file(PROCESSED_PATH, "features_x.npy")
labels_file = create_file(PROCESSED_PATH, 'labels_y.npy')

features, labels = log_mel_spec(AUDIO_PATH,
                                ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9',
                                 'fold10'])
labels_encoded = encode(labels)

np.save(feature_file, features)
np.save(labels_file, labels_encoded)
print("Congratulations! Data processing stage has completed!✨")
