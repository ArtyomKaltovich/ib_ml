import contextlib

import numpy as np
import tensorflow as tf
from common.data import read_notmnist_dataset


def main():
    data, target, labels = read_notmnist_dataset()
    X_train, y_train, X_test, y_test = train_test_split(data, target)
    model = _define_model("relu")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(data, target, epochs=10, batch_size=1000)
    result = model.predict(X_test)
    print(f"Test Accurasy: {np.mean(result.argmax(axis=1) == y_test)}")


def train_test_split(data, target, *, test_frac=0.8):
    train_indices = np.random.choice(a=[False, True], size=len(target), p=[1 - test_frac, test_frac])
    test_indices = ~train_indices
    return data[train_indices], target[train_indices], data[test_indices], target[test_indices]


def _define_model(activation):
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    prev = inputs
    for _ in range(2):
        prev = conv2d_with_skip_connection(activation, prev)
    prev = tf.keras.layers.Flatten()(prev)
    for _ in range(4):
        prev = stack_dense(activation, prev)
    predictions = tf.keras.layers.Dense(10, activation="softmax")(prev)
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    return model


def stack_dense(activation, prev):
    stack1 = tf.keras.layers.Dense(128, activation=activation)(prev)
    stack2 = tf.keras.layers.Dense(128, activation=activation)(prev)
    prev = tf.keras.layers.concatenate([stack1, stack2])
    return prev


def conv2d_with_skip_connection(activation, prev):
    prev1 = tf.keras.layers.Conv2D(8, kernel_size=3, activation=activation)(prev)
    prev2 = tf.keras.layers.Conv2D(8, kernel_size=3, activation=activation, padding="same")(prev1)
    prev = tf.keras.layers.add([prev1, prev2])
    return prev


if __name__ == '__main__':
    with contextlib.redirect_stdout(open(f"result/not_mnist.txt", "w")):
        main()
