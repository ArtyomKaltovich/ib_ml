import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from common.data import read_blobs2

DATASET_PLOT = r"plot/dataset.png"


def visualize_dataset():
    X, y = read_blobs2()
    plt.figure(figsize=(10, 10))
    sns.scatterplot(X["x"], X["y"], hue=y)
    plt.savefig(DATASET_PLOT)


def svm():
    X, y = read_blobs2()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    kernels = dict(linear=[dict(C=0.1), dict(C=0.5), dict(C=1.0), dict(C=2)],
                   poly=[dict(degree=1), dict(degree=2), dict(degree=3), dict(degree=4)],
                   rbf=[dict(C=0.1), dict(C=0.5), dict(C=1.0), dict(C=2)])
    for kernel, params in kernels.items():
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        for param, ax in zip(params, axs.ravel()):
            clr = SVC(kernel=kernel, **param)
            clr.fit(X_train, y_train)
            result = clr.predict(X_test)
            _draw_decision_curve(clr, X, ax)
            _draw_vectors_with_supports(clr, X, y, ax)
            accuracy = accuracy_score(y_test, result)
            param_string = ",".join(f"{key}={value}" for key, value in param.items())
            ax.set_title(f"accuracy={accuracy}, {param_string}")
            plt.tight_layout()
            plt.savefig(f"plot/{kernel}.png")


def _draw_vectors_with_supports(clr, X, y, ax):
    support = np.array(["ordinary"] * len(X))
    np.put(support, clr.support_, "support")
    sns.scatterplot(data=X, x="x", y="y", hue=y, style=support, ax=ax)


def _draw_decision_curve(clr, X, ax):
    step, maxs, mins = _calc_step_max_and_min(X)
    x_min, y_min = mins
    x_max, y_max = maxs
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = clr.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z)


def _calc_step_max_and_min(X):
    X = np.array(X)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    step = max(maxs - mins) / 500
    return step, maxs, mins


if __name__ == '__main__':
    #visualize_dataset()
    svm()
