import numpy as np


class NaiveBayes:
    def __init__(self, alpha=0.01):
        self.alpha = alpha  # Параметр аддитивной регуляризации
        self.word_probs = None
        self.labels = None
        self.label_probs = None

    def fit(self, X, y):
        labels = np.unique(y)
        word_probs = []
        label_probs = []
        for l in labels:
            is_l_class = (y == l)
            objects_of_l_class = X[is_l_class]
            n = objects_of_l_class.sum()
            word_probs.append((sum(objects_of_l_class) + self.alpha) / (n + self.alpha * n))
            label_probs.append(np.mean(is_l_class))
        self.labels = labels
        self.word_probs = np.array(word_probs)
        self.label_probs = np.array(label_probs)

    def predict(self, X):
        return [self.labels[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        return (X + self.alpha) @ self.word_probs.T
