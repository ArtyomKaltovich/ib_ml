import abc
import copy
import operator
import random
from dataclasses import dataclass, astuple

import numpy as np
import scipy


def l1_path_len(path):
    dist = path[:-1] - path[1:]
    return abs(dist).sum()


@dataclass
class Optimization:
    dist: float
    path: np.ndarray

    def __iter__(self):
        return iter(astuple(self))


class AbstractPathOptimizer(abc.ABC):
    def __init__(self, n_iter=10000):
        self.n_iter = n_iter

    @abc.abstractmethod
    def next_step(self, data):
        pass

    def __call__(self, data):
        data = np.array(data)
        min_path_len = None
        min_path = None
        self.init(data)
        for _ in range(self.n_iter):
            data = self.next_step(data)
            if not hasattr(data, "dist"):
                dist = l1_path_len(data)
            else:
                dist, data = data
            if min_path_len is None or min_path_len > dist:
                min_path_len = dist
                min_path = data.copy()
        return Optimization(min_path_len, min_path)

    def init(self, data):
        np.random.shuffle(data)

    def _swap(self, data, idx1, idx2):
        temp = data[idx1].copy()
        data[idx1] = data[idx2]
        data[idx2] = temp

    def _swap_random_pair(self, data):
        assert len(data) > 1
        idx1 = random.randrange(0, len(data))
        idx2 = idx1
        while idx2 == idx1:
            idx2 = random.randrange(0, len(data))
        self._swap(data, idx1, idx2)
        return idx1, idx2

    def _shuffle(self, data):
        data = data.copy()
        np.random.shuffle(data)
        return Optimization(l1_path_len(data), data)


class MonteCarloOptimizer(AbstractPathOptimizer):
    def next_step(self, data):
        np.random.shuffle(data)
        return data


class RandomWalkOptimizer(AbstractPathOptimizer):
    def next_step(self, data):
        idx = random.randrange(0, len(data) - 1)
        self._swap(data, idx, idx + 1)
        return data


class GeneticOptimizer(AbstractPathOptimizer):
    def __init__(self, n_generation=1000, generation_size=100, survival_rate=0.5):
        super().__init__(n_generation)
        self.generation_size = generation_size
        self.survival_rate = survival_rate
        self._generation = None
        self._last_survived_index = None

    def next_step(self, data):
        self._generation = sorted(self._generation, key=operator.attrgetter("dist"))
        print(self._generation[0].dist)
        idx = self._last_survived_index
        for i in range(idx, len(self._generation)):
            parent_idx = i % self._last_survived_index
            _, child = copy.deepcopy(self._generation[parent_idx])
            self._swap_random_pair(child)
            self._generation[i] = Optimization(l1_path_len(child), child)
        return self._generation[0].path

    def init(self, data):
        self._generation = [self._shuffle(data) for _ in range(self.generation_size)]
        self._last_survived_index = int(len(data) * self.survival_rate)


def linear_decrease(n_iteration):
    def func(temp):
        return temp - 1 / n_iteration
    return func


class AnnealingOptimizer(AbstractPathOptimizer):
    def __init__(self, temp_dec_func=linear_decrease(10000)):
        self.temp_dec_func = temp_dec_func
        self._temp = 1
        self._min_path_len = None
        self._min_path = None

    def __call__(self, data):
        data = np.array(data)
        self.init(data)
        while self._temp > 0:
            self.next_step(data)
            self._temp = self.temp_dec_func(self._temp)
        return Optimization(self._min_path_len, self._min_path)

    def init(self, data):
        super().init(data)
        self._temp = 1
        self._min_path_len = l1_path_len(data)
        self._min_path = data.copy()

    def next_step(self, data):
        np.random.shuffle(data)
        dist = l1_path_len(data)
        if random.random() < scipy.special.expit((self._min_path_len - dist) / self._temp / self._min_path_len):
            self._min_path_len = dist
            self._min_path = data.copy()


class HillClimbOptimizer(AbstractPathOptimizer):
    def __init__(self, n_iter=1000, n_climbers=100):
        super().__init__(n_iter)
        self.n_climbers = n_climbers
        self.climbers = None
        self.best_climber = None

    def init(self, data):
        super().init(data)
        self.climbers = [self._shuffle(data) for _ in range(self.n_climbers)]

    def next_step(self, data):
        for climber in self.climbers:
            path = climber.path
            idxs = self._swap_random_pair(path)
            dist = l1_path_len(path)
            if dist < climber.dist:
                climber.dist = dist
            else:
                self._swap(path, *idxs)
            if not self.best_climber or dist < self.best_climber.dist:
                self.best_climber = climber
        return self.best_climber


def find_path(data, optimizer=GeneticOptimizer()):
    return optimizer(data)
