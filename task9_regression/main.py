from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import copy

from common.data import read_cancer_dataset, read_spam_dataset
from task9_regression.regression import mse, NormalLR, GradientLR


def generate_synthetic(size, dim=6, noise=0.1):
    np.random.seed(42)
    X = np.random.randn(size, dim)
    w = np.random.randn(dim + 1)
    noise = noise * np.random.randn(size)
    y = X.dot(w[1:]) + w[0] + noise
    return X, y


def exact_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    regr = NormalLR()
    regr.fit(X_train, y_train)
    return mse(y_test, regr.predict(X_test))


def gradient_regression(X, y, exact_error, file_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    coeffs = [0.0, 0.001, 0.0033, 0.01, 0.33, 0.1]
    errors = []
    for lasso_reg_coef in coeffs:
        regr = GradientLR(lasso_reg_coef=lasso_reg_coef)
        regr.fit(X_train, y_train)
        error = mse(y_test, regr.predict(X_test))
        print(error)
        errors.append(error)
    sns.lineplot(coeffs, errors)
    sns.lineplot(coeffs, [exact_error] * len(coeffs))
    plt.xscale("log")
    plt.xlabel("lasso coefficient")
    plt.ylabel("cost")
    plt.legend(("lasso cost", "exact error"))
    plt.savefig(file_name)
    plt.clf()


if __name__ == '__main__':
    X, y = generate_synthetic(1024)
    error = exact_regression(X, y)
    gradient_regression(X, y, error, "plot/synthetic.png")
    #X, y = read_cancer_dataset()
    #X = X.to_numpy()
    #y = y.to_numpy()
    #error = exact_regression(X, y)
    #gradient_regression(X, y, error, "plot/cancer.png")
    #X, y = read_spam_dataset()
    #X = X.to_numpy()
    #y = y.to_numpy()
    #error = exact_regression(X, y)
    #gradient_regression(X, y, error, "plot/spam.png")
