import numpy as np


class Perceptron:
    def __init__(self, iterations=100, optimal=True):
        self._iterations = iterations
        self.optimal = optimal
        self.w = None
        self.labels = None

    def fit(self, X, y):
        newX = self._preprocess_X(X)
        self.labels, newY = np.unique(y, return_inverse=True)
        assert len(self.labels) == 2
        w = np.random.uniform(-1, 1, size=(newX.shape[1], 1))
        min_n_errors = float("+inf")
        for _ in range(self._iterations):
            n_errors = 0
            for row, target in zip(newX, newY):
                prediction = int(row @ w < 0)
                if target != prediction:
                    w += row.reshape(-1, 1) * (-1 if target else 1)
                    n_errors += 1
            if self.optimal and n_errors < min_n_errors:
                min_n_errors = n_errors
                optimal_w = w.copy()
        self.w = optimal_w if self.optimal else w
        pass

    def predict(self, X):
        newX = self._preprocess_X(X)
        result = np.array(newX @ self.w < 0, dtype=int)
        return self.labels[result]

    def _preprocess_X(self, X):
        newX = self._normalize(X)
        newX = self._copy_X_and_add_column(newX)
        return newX

    def _normalize(self, newX):
        newX /= abs(newX).max(axis=0)
        return newX

    def _copy_X_and_add_column(self, X):
        n, m = X.shape
        newX = np.ones((n, m + 1))
        newX[:, 1:] = X
        return newX
