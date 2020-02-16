import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split as sk_train_test_split


def _read(path_to_csv):
    data = pd.read_csv(path_to_csv)
    data = data.sample(frac=1).reset_index()
    target = data["label"]
    data = data.drop(["label", "index"], axis=1)
    data = (data - data.mean()) / abs(data.max())
    return data, target


def read_cancer_dataset(path_to_csv):
    X, y = _read(path_to_csv)
    y = y.apply(lambda elem: 1 if elem == "M" else 0)
    return X, y

def read_spam_dataset(path_to_csv):
    return _read(path_to_csv)


def train_test_split(X, y, ratio):
    # Возвращает X_train, y_train, X_test, y_test
    # X_train и X_test - массив векторов - две части массива X, разделенного в состветсви с коэффициентом ratio
    # y_train и y_test - соответствующие X_train и X_test метки классов
    X_train, X_test, y_train, y_test = sk_train_test_split(X, y, train_size=ratio)
    return X_train, y_train, X_test, y_test


def get_precision_recall_accuracy(y_pred, y_true):
    # Возвращает precision, recall и accuracy
    # precision - набор значений метрики precision для каждого класса
    # recall - набор значений метрики recall для каждого класса
    # accuracy - число, отражающее общую точность предсказания
    assert len(y_pred) == len(y_true)
    tp, tn, fp, fn = [0] * 4
    for pred, true in zip(y_pred, y_true):
        if pred:
            if true:
                tp += 1
            else:
                fp += 1
        else:
            if true:
                fn += 1
            else:
                tn += 1
    temp = tp + fp
    precision = tp / temp if temp else 0.0
    temp = tn + fn
    precision2 = tn / temp if temp else 0.0
    temp = tp + fn
    recall = tp / temp if temp else 0.0
    temp = tn + fp
    recall2 = tn / temp if temp else 0.0
    temp = (tp + fp + tn + fn)
    accuracy = (tp + tn) / temp if temp else 0.0
    return [precision, precision2], [recall, recall2], accuracy


def plot_precision_recall(X_train, y_train, X_test, y_test, max_k=30):
    ks = list(range(1, max_k + 1))
    classes = len(np.unique(list(y_train) + list(y_test)))
    precisions = [[] for _ in range(classes)]
    recalls = [[] for _ in range(classes)]
    accuracies = []
    for k in ks:
        classifier = KNearest(k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in range(classes):
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(ys) - 0.01, np.max(ys) + 0.01)
        for cls, cls_y in enumerate(ys):
            plt.plot(x, cls_y, label="Class " + str(cls))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    plot(ks, recalls, "Recall")
    plot(ks, precisions, "Precision")
    plot(ks, [accuracies], "Accuracy", legend=False)


def plot_roc_curve(X_train, y_train, X_test, y_test, max_k=30):
    positive_samples = sum(1 for y in y_test if y == 0)
    ks = list(range(1, max_k + 1))
    curves_tpr = []
    curves_fpr = []
    colors = []
    for k in ks:
        colors.append([k / ks[-1], 0, 1 - k / ks[-1]])
        knearest = KNearest(k)
        knearest.fit(X_train, y_train)
        p_pred = [p[0] for p in knearest.predict_proba(X_test)]
        tpr = []
        fpr = []
        for w in np.arange(-0.01, 1.02, 0.01):
            y_pred = [(0 if p > w else 1) for p in p_pred]
            tpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0) / positive_samples)
            fpr.append(
                sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0) / (len(y_test) - positive_samples))
        curves_tpr.append(tpr)
        curves_fpr.append(fpr)
    plt.figure(figsize=(7, 7))
    for tpr, fpr, c in zip(curves_tpr, curves_fpr, colors):
        plt.plot(fpr, tpr, color=c)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()


class KNearest:
    def __init__(self, n_neighbors=5, leaf_size=30):
        self._tree = None
        self._leaf_size = leaf_size
        self._n_neighbors = n_neighbors
        self._point_to_label = {}

    def fit(self, X, y):
        data = copy.copy(X)
        self._y = np.array(y)
        self._tree = KDTree(data, leafsize=self._leaf_size)

    def predict_proba(self, X):
        result, indexes = self._tree.query(X, k=self._n_neighbors)
        result = np.zeros((len(result), 2), float)
        for i, index in enumerate(indexes):
            index = index if isinstance(index, np.ndarray) else (index, )
            proba = sum(self._y.take(index)) / len(index)
            result[i][0] = 1 - proba
            result[i][1] = proba
        return result

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == '__main__':
    #os.chdir("task1_NN")
    X, y = read_cancer_dataset("sample_data/cancer.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    plot_precision_recall(X_train, y_train, X_test, y_test)
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=10)

    X, y = read_spam_dataset("sample_data/spam.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    plot_precision_recall(X_train, y_train, X_test, y_test, max_k=20)
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=20)
