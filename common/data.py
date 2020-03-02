import pandas as pd


def _read(path_to_csv):
    data = pd.read_csv(path_to_csv)
    data = data.sample(frac=1).reset_index()
    target = data["label"]
    data = data.drop(["label", "index"], axis=1)
    data = (data - data.mean()) / abs(data.max())
    return data, target


def read_cancer_dataset(path_to_csv=r"../sample_data/cancer.csv"):
    X, y = _read(path_to_csv)
    y = y.apply(lambda elem: 1 if elem == "M" else 0)
    return X, y


def read_spam_dataset(path_to_csv=r"../sample_data/spam.csv"):
    return _read(path_to_csv)
