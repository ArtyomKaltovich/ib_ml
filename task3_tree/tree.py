import itertools

import numpy as np
import pandas as pd

from task3_tree.criterion import gini, gain


class DecisionTreeNode:
    def __init__(self, parent, split_dim, split_value, left_data, right_data):
        self.parent = parent
        self.split_dim = split_dim
        self.split_value = split_value
        self.left_data = left_data
        self.right_data = right_data
        self.left_node = None
        self.right_node = None

    def step(self, X, y, mask):
        klass = X[:, self.split_dim] < self.split_value
        self.left_node.step(X, y, mask & klass)
        self.right_node.step(X, y, mask & ~klass)
        return y


class DecisionTreeLeaf:
    def __init__(self, parent, y=None, mask=None):
        self.parent = parent
        self.y = y
        self.proba = sum(y[mask]) / sum(mask)

    def step(self, X, y, mask):
        y[mask] = self.proba


class DecisionTreeClassifier:
    def __init__(self, criterion=gini, max_depth=None, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            mask = np.ones(len(X), dtype=bool)
            X = np.array(X)
            split_column, split_value, split_left, split_right = self.step(X, y, mask)
            self.root = DecisionTreeNode(self, split_column, split_value, split_left, split_right)
            # format: parent, node_mask, is_right, height
            stack = [(self.root, split_left, 0, 1), (self.root, split_right, 1, 1)]
            while stack:
                self._add_node(*stack.pop(), X, y, stack)
        else:
            raise NotImplementedError("only pandas DataFrames supported ax X for now")

    def _add_node(self, parent, mask, is_right, height, X, y, stack):
        masked = y[mask]
        if len(masked) == 0:
            node = DecisionTreeLeaf(self)
        if sum(masked) == 0 or sum(masked) == sum(mask) or len(masked) <= self.min_samples_leaf\
                or height >= self.max_depth:
            node = DecisionTreeLeaf(self, y, mask)
        else:
            split_column, split_value, split_left, split_right = self.step(X, y, mask)
            node = DecisionTreeNode(self, split_column, split_value, split_left, split_right)
            stack.append((node, split_left, 0, height + 1))
            stack.append((node, split_right, 1, height + 1))
        if is_right:
            parent.right_node = node
        else:
            parent.left_node = node

    def step(self, X, y, mask):
        class_zero = X[mask & (y == 0)]
        class_one = X[mask & (y == 1)]
        means = range(X.shape[1]), np.mean(class_zero, axis=0), np.mean(class_one, axis=0)
        medians = range(X.shape[1]), np.median(class_zero, axis=0), np.median(class_one, axis=0)
        candidates = itertools.chain(zip(*means), zip(*medians))
        min_gain = float("inf")
        split_value = None
        split_column = None
        split_left = None
        split_right = None
        for col, val0, val1 in candidates:
            val = (val0 + val1) / 2
            left_y = mask & (X[:, col] < val)
            right_y = mask & (X[:, col] >= val)
            if any(left_y) and any(right_y):  # sometimes there are no elements below(or above) mean and (EVEN!) median
                current = gain(left_y, right_y, self.criterion)
                if current < min_gain:
                    min_gain = current
                    split_value = val
                    split_column = col
                    split_left = left_y
                    split_right = right_y
        return split_column, split_value, split_left, split_right

    def predict_proba(self, X):
        y = np.zeros(len(X))
        mask = np.ones(len(X), dtype=bool)
        return self.root.step(np.array(X), y, mask)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array(proba > 0.5, dtype=int)
