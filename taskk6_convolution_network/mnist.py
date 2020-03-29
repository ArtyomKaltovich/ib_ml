import contextlib

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from common.data import read_mnist_dataset


def main(activation):
    X_test, X_train, y_test, y_train = _load_test_train_for_keras()
    model = _define_model(activation)
    result = fit_predict(X_test, X_train, model, y_train)
    print(f"Test Accurasy: {np.mean(result.argmax(axis=1) == y_test)}")


def fit_predict(X_test, X_train, model, y_train):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=1000)
    result = model.predict(X_test)
    return result


def _load_test_train_for_keras():
    data, target = read_mnist_dataset()
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.8, shuffle=False)
    return X_test, X_train, y_test, y_train


def _define_model(activation):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, kernel_size=3, activation=activation),
        tf.keras.layers.Conv2D(8, kernel_size=3, activation=activation),
        tf.keras.layers.Conv2D(8, kernel_size=3, activation=activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model


if __name__ == '__main__':
    for activation in ["relu", "sigmoid", "tanh"]:
        with contextlib.redirect_stdout(open(f"result/first_part_{activation}.txt", "w")):
            main(activation)
