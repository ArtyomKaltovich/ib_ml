import heapq
import itertools
from functools import partial, update_wrapper

import numpy as np

from scipy.spatial.distance import euclidean
from sklearn.neighbors._kd_tree import KDTree


def average():
    def func(*elems):  # create new function to save np.mean data
        return np.mean(*elems)
    update_wrapper(func, average)
    return func


def single():
    def func(*elems):  # create new function, because min is unwrapable
        return min(*elems)
    update_wrapper(func, single)
    return func


def complete():
    def func(*elems):
        return max(*elems)
    update_wrapper(func, complete)
    return func


class AgglomertiveClustering:
    def __init__(self, n_clusters=16, linkage=average()):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def metric(self, X, Y, n_features=None, dist_func=euclidean):
        small, big = (X, Y) if len(X) > len(Y) else (Y, X)
        small = small.reshape(-1, n_features)
        big = big.reshape(-1, n_features)
        kdtree = KDTree(big)
        result, _ = kdtree.query(small)
        result = self.linkage(result)
        return result

    def fit_predict(self, X, y=None):
        metric = partial(self.metric, n_features=X.shape[1])
        to_unite = set((i,) for i in range(len(X)))
        distances = [(metric(X[i], X[j]), (i,), (j,)) for i, j in itertools.combinations(range(len(X)), r=2)]
        heapq.heapify(distances)
        while len(to_unite) > self.n_clusters:
            mate1, mate2 = self._select_mates(distances, to_unite)
            self._recalculate_dists(X, distances, mate1, mate2, metric, to_unite)
        result = np.zeros(len(X), dtype=int)
        for number, points in enumerate(to_unite):
            result.put(points, [number] * len(points))
        return result

    def _recalculate_dists(self, X, distances, mate1, mate2, metric, to_unite):
        to_unite.discard(mate1)
        to_unite.discard(mate2)
        new_mate = mate1 + mate2
        for c in to_unite:
            new_item = (metric(X.take(c, axis=0), X.take(new_mate, axis=0)), new_mate, c)
            heapq.heappush(distances, new_item)
        to_unite.add(new_mate)

    def _select_mates(self, distances, to_unite):
        dist, mate1, mate2 = heapq.heappop(distances)
        while mate1 not in to_unite or mate2 not in to_unite:
            dist, mate1, mate2 = heapq.heappop(distances)
        return mate1, mate2
