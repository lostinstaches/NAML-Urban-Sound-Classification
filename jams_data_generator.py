from typing import Any

import numpy as np
from keras.utils import to_categorical, Sequence
import os
import jams
import muda


class JamsDataGenerator(Sequence):
    def __init__(
        self,
        fold_path: str,
        modifications_to_include: list[str] = [
            "bgnoise",
            "pitch1",
            "pitch2",
            "stretch",
            "drc",
            "original",
        ],
        audio_file_extension: str = ".wav",
        mini_batch_size: int = 64,
        use_categorical_labels: bool = True,
        shuffle_samples: bool = True,
    ):
        self.mini_batch_size = mini_batch_size
        self.use_categorical_labels = use_categorical_labels
        self.shuffle_samples = shuffle_samples
        self.samples_with_metadata: dict[str, dict[str, Any]] = {}
        self.samples_names: list[str] = []

        for modification in filter(
            lambda p: p in modifications_to_include, os.listdir(fold_path)
        ):
            path_to_jams_files = os.path.join(fold_path, modification, "jams")
            path_to_audio_files = os.path.join(fold_path, modification, "audio")
            for jams_file_name in sorted(
                filter(
                    lambda file_path: file_path.endswith(".jams"),
                    os.listdir(path_to_jams_files),
                )
            ):
                jams_file = jams.load(f"{path_to_jams_files}/{jams_file_name}")
                metadata = jams_file.annotations[0]["sandbox"]
                sample_name = jams_file_name[:-5]
                self.samples_names.append(sample_name)
                self.samples_with_metadata[sample_name] = {
                    "jams_file": f"{path_to_jams_files}/{jams_file_name}",
                    "audio_file": f"{path_to_audio_files}/{sample_name}{audio_file_extension}",
                    "class": metadata["class"],
                    "class_id": metadata["classID"],
                }

        self.num_classes = max(
            sample["class_id"] + 1 for sample in self.samples_with_metadata.values()
        )

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return len(self.samples_names) // self.mini_batch_size

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        mini_batch_sample_names = self.samples_names[
            index * self.mini_batch_size : (index + 1) * self.mini_batch_size
        ]

        # Generate data
        samples = [
            muda.load_jam_audio(
                self.samples_with_metadata[sample]["jams_file"],
                self.samples_with_metadata[sample]["audio_file"],
            )
            for sample in mini_batch_sample_names
        ]
        labels = [
            self.samples_with_metadata[sample]["class_id"]
            for sample in mini_batch_sample_names
        ]
        if self.use_categorical_labels:
            labels = to_categorical(labels, num_classes=self.num_classes)

        return samples, labels

    def on_epoch_end(self):
        if self.shuffle_samples:
            np.random.shuffle(self.samples_names)
