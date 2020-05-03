import abc

import numpy as np
import scipy


def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


class AbstractLR(abc.ABC):
    def __init__(self):
        self.w = None

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    def _preprocess_X(self, X):
        #X = self._normalize(X)
        X = self._copy_X_and_add_column(X)
        return X

    def _normalize(self, X):
        X = X / abs(X).max(axis=0)
        return X

    def _copy_X_and_add_column(self, X):
        n, m = X.shape
        newX = np.ones((n, m + 1))
        newX[:, 1:] = X
        return newX


class NormalLR(AbstractLR):
    def fit(self, X, y):
        X = self._preprocess_X(X)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = self._preprocess_X(X)
        return X @ self.w


class GradientLR(AbstractLR):
    def __init__(self, learning_rate=0.001, iterations=100, lasso_reg_coef=0.0, grad_step=0.001, batch_ratio=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lasso_reg_coef = lasso_reg_coef
        self.step = grad_step
        self._2step = 2 * self.step
        self.batch_ratio = batch_ratio

    def fit(self, X, y):
        X = self._preprocess_X(X)
        w = np.random.uniform(-1, 1, size=(X.shape[1]))
        for i in range(self.iterations*10):
            sample = np.random.random(len(X)) < self.batch_ratio
            sample_x = X[sample]
            sample_y = y[sample]
            error = self._cost(sample_x, w, sample_y)
            grad = scipy.optimize.approx_fprime(w, lambda w: self._cost(sample_x, w, sample_y), self.step)
            w -= self.learning_rate * grad * error
            #if not (i % 500):
            #c    print(error)
        self.w = w

    def _cost(self, X, w, y):
        return mse(y, X @ w) + self.lasso_reg_coef * sum(abs(w))

    def predict(self, X):
        X = self._preprocess_X(X)
        return X @ self.w
