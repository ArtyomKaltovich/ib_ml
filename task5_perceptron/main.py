import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
import copy
from perceptron import Perceptron

BLOBS_PLOT = "plot/blobs.png"
MOONS_PLOT = "plot/moons.png"
DIGITS_PLOT = "plot/digits_1_and_5.png"


def visualize(X, labels_true, labels_pred, w, path):
    unique_labels = np.unique(labels_true)
    unique_colors = dict([(l, c) for l, c in zip(unique_labels, [[0.8, 0., 0.], [0., 0., 0.8]])])
    plt.figure(figsize=(9, 9))

    if w[1] == 0:
        plt.plot([X[:, 0].min(), X[:, 0].max()], w[0] / w[2])
    elif w[2] == 0:
        plt.plot(w[0] / w[1], [X[:, 1].min(), X[:, 1].max()])
    else:
        mins, maxs = X.min(axis=0), X.max(axis=0)
        pts = [[mins[0], -mins[0] * w[1] / w[2] - w[0] / w[2]],
               [maxs[0], -maxs[0] * w[1] / w[2] - w[0] / w[2]],
               [-mins[1] * w[2] / w[1] - w[0] / w[1], mins[1]],
               [-maxs[1] * w[2] / w[1] - w[0] / w[1], maxs[1]]]
        pts = np.array(pts)
        pts = [(x, y) for x, y in pts if mins[0] <= x <= maxs[0] and mins[1] <= y <= maxs[1]]
        x, y = list(zip(*pts))
        plt.plot(x, y, c=(0.75, 0.75, 0.75), linestyle="--")

    colors_inner = [unique_colors[l] for l in labels_true]
    colors_outer = [unique_colors[l] for l in labels_pred]
    plt.scatter(X[:, 0], X[:, 1], c=colors_inner, edgecolors=colors_outer)
    #plt.show()
    plt.savefig(path)


def blobs():
    X, true_labels = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5]])
    c = Perceptron()
    c.fit(X, true_labels)
    pred = c.predict(X).reshape(-1)
    visualize(X, true_labels, pred, c.w.reshape(-1), BLOBS_PLOT)


def moons():
    X, true_labels = make_moons(400, noise=0.075)
    c = Perceptron()
    c.fit(X, true_labels)
    pred = c.predict(X).reshape(-1)
    visualize(X, true_labels, pred, c.w.reshape(-1), MOONS_PLOT)


def transform_images(images):
    pca = PCA(n_components=2)
    return pca.fit_transform(images.reshape(len(images), -1))
    mean = np.mean(images.sum(1), axis=(1))
    symm = np.mean(images[:, ::-1, :] * images, axis=(1, 2))
    result = np.stack((mean, symm))
    result = result.reshape(-1, 2)
    return result


def get_digits(y0=1, y1=5):
    data = datasets.load_digits()
    images, labels = data.images, data.target
    mask = np.logical_or(labels == y0, labels == y1)
    labels = labels[mask]
    images = images[mask]
    X = transform_images(images)
    #visualize_transformed(X, labels)
    return X, labels


def visualize_transformed(X, labels):
    X = X / abs(X).max(axis=0)
    colors_inner = [(1, 0, 0) if x == 1 else (0, 0, 1) for x in labels]
    plt.scatter(X[:, 0], X[:, 1], c=colors_inner)
    plt.show()


def one_vs_five():
    X, y = get_digits()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    c = Perceptron(iterations=100)
    c.fit(X_train, y_train)
    pred = c.predict(X_train).reshape(-1)
    visualize(X_train, y_train, pred, c.w.reshape(-1), DIGITS_PLOT)
    print("Accuracy:", np.mean(pred == y_test))


if __name__ == '__main__':
    #one_vs_five()
    accs = []
    for optimal in (False, True):
        for y0, y1 in [(y0, y1) for y0 in range(9) for y1 in range(y0 + 1, 10)]:
            X, y = get_digits(y0, y1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
            c = Perceptron(iterations=200)
            c.fit(X_train, y_train)
            pred = c.predict(X_test).reshape(-1)
            accs.append(np.mean(pred == y_test))
        print(f"Mean accuracy: {np.mean(accs)}, for {'with pocket' if optimal else 'without pocket'}")
