from contextlib import contextmanager
from functools import partialmethod, partial
from time import perf_counter_ns

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from common.data import read_cancer_dataset, read_spam_dataset

N_ITERATIONS = 25


def svm_vs_random_forest(dataset):
    X, y = read_dataset(dataset)
    measurements = compare_clf(X, y)
    visualize(dataset, measurements)


def visualize(dataset, measurements):
    sns.stripplot(data=measurements, x="Classifier", y="Accuracy")
    plt.tight_layout()
    plt.savefig(f"plot/{dataset}_accuracy.png")
    plt.clf()
    sns.stripplot(data=measurements, x="Classifier", y="Fit Time(ms)")
    plt.tight_layout()
    plt.savefig(f"plot/{dataset}_fit_time.png")
    plt.clf()
    sns.stripplot(data=measurements, x="Classifier", y="Predict Time (ms)")
    plt.tight_layout()
    plt.savefig(f"plot/{dataset}_predict_time.png")
    plt.clf()


def read_dataset(dataset):
    if dataset == "cancer":
        X, y = read_cancer_dataset()
    elif dataset == "spam":
        X, y = read_spam_dataset()
    else:
        raise ValueError("Please choose cancer or spam dataset")
    return X, y


def compare_clf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    columns = "Classifier", "Accuracy", "Fit Time(ms)", "Predict Time (ms)"
    measurements = pd.DataFrame(columns=columns)
    for idx, (name, constructor) in enumerate((
            ("SVC", partial(SVC, random_state=42)),
            ("Random forest, threads=1", partial(RandomForestClassifier, random_state=42, n_jobs=1)),
            ("Random forest, threads=4", partial(RandomForestClassifier, random_state=42, n_jobs=4)))):
        accuracy = []
        fit_time = []
        predict_time = []
        for _ in range(N_ITERATIONS):
            clf = constructor()
            with benchmark(fit_time):
                clf.fit(X_train, y_train)
            with benchmark(predict_time):
                result = clf.predict(X_test)
            accuracy.append(accuracy_score(y_test, result))
        currents = pd.DataFrame([[name] * N_ITERATIONS, accuracy, fit_time, predict_time]).T
        currents.columns = columns
        measurements = measurements.append(currents)
    return measurements


@contextmanager
def benchmark(measurements):
    start = perf_counter_ns()
    yield
    measurements.append((perf_counter_ns() - start) / 1_000_000)


if __name__ == '__main__':
    svm_vs_random_forest("spam")
    svm_vs_random_forest("cancer")
