import os
import pandas as pd
import numpy as np
from matplotlib.image import imread


def _read(path_to_csv, *, normalize=True):
    if path_to_csv.endswith(".csv"):
        data = pd.read_csv(path_to_csv)
    elif path_to_csv.endswith(".gz"):
        data = pd.read_csv(path_to_csv, compression="gzip")
    else:
        raise ValueError("Unknown file extension, please use csv or gz files")
    data = data.sample(frac=1).reset_index()
    target = data["label"]
    data = data.drop(["label", "index"], axis=1)
    if normalize:
        not_all_zero_columns = data.columns[abs(data.max()) != 0]
        data[not_all_zero_columns] = (data[not_all_zero_columns] - data[not_all_zero_columns].mean())\
                                     / abs(data[not_all_zero_columns].max())
    return data, target


def read_cancer_dataset(path_to_csv=r"../sample_data/cancer.csv"):
    X, y = _read(path_to_csv)
    y = y.apply(lambda elem: 1 if elem == "M" else 0)
    return X, y


def read_spam_dataset(path_to_csv=r"../sample_data/spam.csv"):
    return _read(path_to_csv)


def read_mnist_dataset(path_to_csv=r"../sample_data/mnist.csv"):
    data, target =_read(path_to_csv, normalize=False)
    data = np.array(data, dtype=float).reshape((10000, 28, 28, 1))
    target = np.array(target)
    return data, target


def read_notmnist_dataset(path_to_folder=r"../sample_data/notMNIST_small"):
    folder = os.walk(path_to_folder)
    _, labels, _ = next(folder)
    labels_to_int = {label: idx for idx, label in enumerate(labels)}
    int_target = []
    data = []
    for parent, _, files in folder:
        for file in files:
            try:
                file = os.path.join(parent, file)
                image = imread(file)
                label = parent[-1]
                int_target.append(labels_to_int[label])
                data.append(image)
            except OSError:
                print(f"Can't read {file}")
    return np.array(data).reshape((-1, 28, 28, 1)), np.array(int_target), np.array(labels)


def read_blobs2(path_to_csv=r"../sample_data/blobs2.csv"):
    return _read(path_to_csv)
