import numpy as np


class NaiveBayes:
    def __init__(self, alpha=0.01):
        self.alpha = alpha  # Параметр аддитивной регуляризации
        self.word_log_probs = None
        self.labels = None
        self.label_log_probs = None

    def fit(self, X, y):
        labels = np.unique(y)
        word_probs = []
        label_probs = []
        for l in labels:
            is_l_class = (y == l)
            objects_of_l_class = X[is_l_class]
            feature_counts = sum(objects_of_l_class) + self.alpha
            word_probs.append(feature_counts / feature_counts.sum())
            label_probs.append(np.mean(is_l_class))
        self.labels = labels
        self.word_log_probs = np.log(word_probs).T
        self.label_log_probs = np.log(label_probs)

    def predict(self, X):
        return [self.labels[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba(self, X):
        return X @ self.word_log_probs + self.label_log_probs
