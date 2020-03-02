import numpy as np
import math


def gini(x):
    if len(x) > 0:  # __bool__ works differently for pandas DataFrame
        nprob, prob = _get_probs(x)
        return 2 * prob * nprob
    return 0


def entropy(x):
    if len(x):  # __bool__ works differently for pandas DataFrame
        nprob, prob = _get_probs(x)
        if not nprob or not prob:
            return 0
        return - (prob * math.log(nprob, 2) + nprob * math.log(prob, 2))
    return 0


def gain(left_y, right_y, criterion):
    l = criterion(left_y)
    r = criterion(right_y)
    n = len(left_y) + len(right_y)
    return (l * len(left_y) + r * len(right_y)) / n


def _get_probs(x):
    prob = sum(x) / len(x)
    nprob = 1 - prob
    return nprob, prob
