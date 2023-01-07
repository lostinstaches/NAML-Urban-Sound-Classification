import os

import numpy as np
import matplotlib.pyplot as plt


def load_data(
    dataset_path: str,
    validation_fold: int,
    modifications_to_include: list[str] = [
        "original",
        "bgnoise",
        "pitch1",
        "pitch2",
        "stretch",
        "drc",
    ],
):
    train_data = None
    train_labels = None
    validation_data = None
    validation_labels = None
    for fold in range(1, 11):
        fold_path = f"{dataset_path}/fold{fold}"
        for modification in os.listdir(fold_path):
            if modification not in modifications_to_include:
                continue
            path_to_images = f"{fold_path}/{modification}/images.npy"
            path_to_labels = f"{fold_path}/{modification}/labels.npy"
            if fold == validation_fold:
                validation_data = (
                    np.concatenate((validation_data, np.load(path_to_images)))
                    if validation_data is not None
                    else np.load(path_to_images)
                )
                validation_labels = (
                    np.concatenate((validation_labels, np.load(path_to_labels)))
                    if validation_labels is not None
                    else np.load(path_to_labels)
                )
            else:
                train_data = (
                    np.concatenate((train_data, np.load(path_to_images)))
                    if train_data is not None
                    else np.load(path_to_images)
                )
                train_labels = (
                    np.concatenate((train_labels, np.load(path_to_labels)))
                    if train_labels is not None
                    else np.load(path_to_labels)
                )
    train_data = train_data.reshape(train_data.shape[0], 128, 130, 1)
    validation_data = validation_data.reshape(validation_data.shape[0], 128, 130, 1)
    return train_data, train_labels, validation_data, validation_labels


def plot_results(history):
    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.grid(linestyle="--")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")

    plt.subplot(122)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.grid(linestyle="--")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")

    plt.show()

    max_acc_epoch = np.argmax(history.history["val_accuracy"])
    min_loss_epoch = np.argmin(history.history["val_loss"])
    print(
        f"Epoch {max_acc_epoch} has the highest validation accuracy: accuracy = {history.history['val_accuracy'][max_acc_epoch]} loss = {history.history['val_loss'][max_acc_epoch]}"
    )
    print(
        f"Epoch {min_loss_epoch} has the lowest validation loss: accuracy = {history.history['val_accuracy'][min_loss_epoch]} loss = {history.history['val_loss'][min_loss_epoch]}"
    )
