from collections import Counter

import numpy as np
from nltk.stem.snowball import SnowballStemmer


class BoW:
    def __init__(self, X, voc_limit=1000):
        self.voc_limit = voc_limit
        self.counts = self.count(X)

    def transform(self, X):
        counts = self.counts
        X = self._preprocess_X(X)
        result = []
        for row in X:
            current_result = []
            counted_row = Counter(row)
            for word in counts:
                current_result.append(counted_row[word])
            result.append(current_result)
        return np.array(result)

    def _preprocess_X(self, X):
        return np.char.split(X)

    def count(self, X):
        X = self._preprocess_X(X)
        counter = Counter()
        for row in X:
            counter.update(row)
        return Counter(dict(counter.most_common(self.voc_limit)))


class BowStem(BoW):
    def __init__(self, X, voc_limit=1000, lang="english"):
        self.stemmer = SnowballStemmer(lang)
        super().__init__(X, voc_limit)

    def _preprocess_X(self, X):
        X = np.char.split(X)
        result = []
        stem = self.stemmer.stem
        for row in X:
            result.append([stem(x) for x in row])
        return result
