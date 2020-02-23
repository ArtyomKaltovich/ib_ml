import random
from functools import partial

import numpy as np
from sklearn.neighbors import KDTree


class Initializations:
    @staticmethod
    def random(start=-10.0, finish=10.0):
        def func(estimator, X):
            random_vector = partial(np.random.uniform, start, finish, size=X.shape[1])
            centroids2 = np.array([random_vector() for _ in range(estimator._n_clusters)])
            return centroids2
        return func

    @staticmethod
    def sample():
        def func(estimator, X):
            centroids2 = np.array(random.sample(list(range(len(X))), k=estimator._n_clusters))  # select indexes
            centroids2 = X[centroids2]
            return centroids2
        return func

    @staticmethod
    def kmeans_plus_plus(metric="minkowski"):
        def func(estimator, X):
            centroids = [random.choice(X)]
            while len(centroids) < estimator._n_clusters:
                kdtree = KDTree(np.array(centroids), metric=metric)  # sklearn's KDTree doesn't take lists
                distances, cluster = kdtree.query(X)
                centroids.append(random.choices(X, weights=distances ** 2)[0])
            return np.array(centroids)

        return func


class KMeans:
    def __init__(self, n_clusters, init=Initializations.kmeans_plus_plus(), max_iter=300, last_shift_norm=0.1):
        self._n_clusters = n_clusters
        self._init = init
        self._max_iter = max_iter
        self.last_shift_norm = last_shift_norm
        self._kdtree = None
        self._centroids = None

    def fit(self, X, y=None):
        centroids2 = self._init(self, X)
        n_iter = 0
        while not n_iter or np.linalg.norm(centroids - centroids2) > self.last_shift_norm:
            centroids = centroids2
            kdtree = KDTree(centroids)
            clusters_sizes = [1] * self._n_clusters
            clusters_sums = centroids.copy()

            dists, clusters = kdtree.query(X)
            for c in range(self._n_clusters):
                mask = (clusters == c).reshape(-1)
                mask = X[mask]
                clusters_sizes[c] += len(mask)
                clusters_sums[c] += sum(mask)
            centroids2 = np.array([sums / size for sums, size in zip(clusters_sums, clusters_sizes)])
            n_iter += 1
            if n_iter >= self._max_iter:
                Warning("iteration number excited")
                break
        self._kdtree = kdtree
        self._centroids = centroids2

    def predict(self, X):
        dists, clusters = self._kdtree.query(X)
        return clusters.reshape(-1)
