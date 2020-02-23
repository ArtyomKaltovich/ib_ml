import cProfile
import pstats
from pstats import SortKey

from sklearn.datasets import make_moons, make_blobs

from task2_Cluster.agglomerate import AgglomertiveClustering
from task2_Cluster.data import visualize_clasters
from task2_Cluster.dbscan import DBScan
from task2_Cluster.k_means import KMeans


def gen_data(visualize=False):
    X_1, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
    X_2, true_labels = make_moons(400, noise=0.075)
    if visualize:
        visualize_clasters(X_1, true_labels)
        visualize_clasters(X_2, true_labels)
    return X_1, X_2


def run_kmeans(X_1, X_2 ):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X_1)
    labels = kmeans.predict(X_1)
    visualize_clasters(X_1, labels)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_2)
    labels = kmeans.predict(X_2)
    visualize_clasters(X_2, labels)


def run_dbscan(X_1, X_2 ):
    dbscan = DBScan(eps=0.8, min_samples=10)
    labels = dbscan.fit_predict(X_1)
    visualize_clasters(X_1, labels)

    dbscan = DBScan(eps=0.15, min_samples=3)
    labels = dbscan.fit_predict(X_2)
    visualize_clasters(X_2, labels)


def run_agglomerate(X_1, X_2):
    agg_clustering = AgglomertiveClustering(n_clusters=4)
    labels = agg_clustering.fit_predict(X_1)
    visualize_clasters(X_1, labels)

    agg_clustering = AgglomertiveClustering(n_clusters=2)
    labels = agg_clustering.fit_predict(X_2)
    visualize_clasters(X_2, labels)


if __name__ == '__main__':
    X_1, X_2 = gen_data()

    #run_kmeans(X_1, X_2)
    #run_dbscan(X_1, X_2)

    pr = cProfile.Profile()
    pr.enable()
    run_agglomerate(X_1, X_2)
    pr.disable()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats()
