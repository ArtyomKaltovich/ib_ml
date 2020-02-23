from sklearn.datasets import make_moons, make_blobs

from task2_Cluster.data import visualize_clasters
from task2_Cluster.k_means import KMeans

X_1, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
#visualize_clasters(X_1, true_labels)
X_2, true_labels = make_moons(400, noise=0.075)
#visualize_clasters(X_2, true_labels)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_1)
labels = kmeans.predict(X_1)
visualize_clasters(X_1, labels)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_2)
labels = kmeans.predict(X_2)
visualize_clasters(X_2, labels)
