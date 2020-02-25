from sklearn.datasets import make_moons, make_blobs

from task2_Cluster import agglomerate
from task2_Cluster.agglomerate import AgglomertiveClustering
from task2_Cluster.data import visualize_clasters
from task2_Cluster.dbscan import DBScan
from task2_Cluster.k_means import KMeans, Initializations


def gen_data(visualize=False):
    X_1, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
    X_2, true_labels = make_moons(400, noise=0.075)
    if visualize:
        visualize_clasters(X_1, true_labels)
        visualize_clasters(X_2, true_labels)
    return X_1, X_2


def run_kmeans(X_1, X_2 ):
    kmeans = KMeans(n_clusters=4)
    for method in (Initializations.random(), Initializations.sample(), Initializations.kmeans_plus_plus()):
        kmeans.fit(X_1)
        labels = kmeans.predict(X_1)
        visualize_clasters(X_1, labels, file_path=f"plot/kmeans_{method.__name__}_1.png")
    kmeans = KMeans(n_clusters=2)
    for method in (Initializations.random(), Initializations.sample(), Initializations.kmeans_plus_plus()):
        kmeans.fit(X_2)
        labels = kmeans.predict(X_2)
        visualize_clasters(X_2, labels, file_path=f"plot/kmeans_{method.__name__}_2.png")


def run_dbscan(X_1, X_2 ):
    for metric, eps, min_samples in (("euclidean", 0.77, 10), ("manhattan", 0.75, 12), ("chebyshev", 0.75, 12)):
        dbscan = DBScan(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(X_1)
        visualize_clasters(X_1, labels, file_path=f"plot/dbscan_{metric}_1.png")

    for metric, eps, min_samples in (("euclidean", 0.2, 3), ("manhattan", 0.2, 4), ("chebyshev", 0.2, 5)):
        dbscan = DBScan(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(X_2)
        visualize_clasters(X_2, labels, file_path=f"plot/dbscan_{metric}_2.png")


def run_agglomerate(X_1, X_2):
    for link in agglomerate.average(), agglomerate.complete(), agglomerate.single():
        agg_clustering = AgglomertiveClustering(n_clusters=4, linkage=link)
        labels = agg_clustering.fit_predict(X_1)
        visualize_clasters(X_1, labels, file_path=f"plot/agglomerate_{link.__name__}_1.png")

        agg_clustering = AgglomertiveClustering(n_clusters=2, linkage=link)
        labels = agg_clustering.fit_predict(X_2)
        visualize_clasters(X_2, labels, file_path=f"plot/agglomerate_{link.__name__}_2.png")


if __name__ == '__main__':
    X_1, X_2 = gen_data()

    #run_kmeans(X_1, X_2)
    #run_dbscan(X_1, X_2)
    run_agglomerate(X_1, X_2)
