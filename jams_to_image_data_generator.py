import librosa
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from jams_data_generator import JamsDataGenerator


class JamsToImageDataGenerator(JamsDataGenerator):
    def __init__(
        self,
        fold_path: str,
        audio_file_extension: str = ".wav",
        mini_batch_size: int = 64,
        audio_sample_rate: int = 44100,
        fft_window_size: int = 1024,
        hop_length: int = 1024,
    ):
        super().__init__(
            fold_path,
            audio_file_extension,
            mini_batch_size,
        )
        self.audio_sample_rate = audio_sample_rate
        self.fft_window_size = fft_window_size
        self.hop_length = hop_length
        self.image_generator = ImageDataGenerator()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return super().__len__()

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        samples, labels = super().__getitem__(index)

        samples_spectograms = []
        for sample, label in zip(samples, labels):
            audio_time_series = sample.sandbox.muda._audio["y"]
            sampling_rate = sample.sandbox.muda._audio["sr"]

            input_patch_length = 3 * self.audio_sample_rate
            if len(audio_time_series) < input_patch_length:
                audio_time_series = np.pad(
                    audio_time_series,
                    (0, input_patch_length - len(audio_time_series)),
                    "wrap",
                )
            audio_time_series = audio_time_series[:input_patch_length]

            # Convert a power spectrogram (amplitude squared) to decibel (dB) units
            spectogram_db = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=audio_time_series,
                    sr=sampling_rate,
                    n_fft=self.fft_window_size,
                    hop_length=self.hop_length,
                )
            )

            samples_spectograms.append(spectogram_db)

        return np.array(samples_spectograms), np.array(labels, dtype=np.int32)
