import numpy as np
from sklearn.neighbors import KDTree


def shrink_cluster_indexes(clusters):
    """ returns new cluster indexes which is 0 to n_unique_elements mapped

    :param clusters:
    :return:

    >>> import numpy as np
    >>> clusters = np.array([0, 0, 1, 3, 10])
    >>> shrink_cluster_indexes(clusters)
    array([0, 0, 1, 2, 3])
    """
    unique, indices = np.unique(clusters, return_inverse=True)
    unique = np.arange(len(unique))
    result = unique[indices]
    return result


class DBScan:
    def __init__(self, eps=0.5, min_samples=5, leaf_size=40, metric="euclidean", max_iter=100):
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric
        self.max_iter = max_iter

    def fit_predict(self, X, y=None):
        kdtree = KDTree(X, metric=self.metric, leaf_size=self.leaf_size)
        clusters = np.arange(len(X))
        n_iter = 0
        while not n_iter or np.linalg.norm(clusters2 - clusters) > 0.1:
            clusters2 = clusters.copy()
            neighbors = kdtree.query_radius(X, self.eps)
            for n in neighbors:
                if len(n) > self.min_samples:
                    clusters[n] = min(clusters[n])
            n_iter += 1
            if n_iter >= self.max_iter:
                Warning("iteration number excited")
                break
        result = shrink_cluster_indexes(clusters)
        return result