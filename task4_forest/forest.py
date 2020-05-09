import numpy as np

from task3_tree.criterion import gini
from task3_tree.tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, criterion=gini, max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.estimators = [DecisionTreeClassifier(criterion, max_depth, min_samples_leaf) for _ in range(n_estimators)]

    def fit(self, X, y):
        for est in self.estimators:
            indices = np.random.randint(0, len(X), size=len(X))
            est.fit(X.take(indices, axis=0), y.take(indices))

    def predict_proba(self, X):
        result = np.zeros((self.n_estimators, len(X)))
        for idx, est in enumerate(self.estimators):
            result[idx] = est.predict_proba(X)
        return result.mean(axis=0)

    def predict(self, X):
        return np.array(self.predict_proba(X) > 0.5, dtype=int)
