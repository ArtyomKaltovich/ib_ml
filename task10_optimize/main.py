import seaborn as sns
import matplotlib.pyplot as plt

from common.data import read_tsp
from task10_optimize.optimize import (find_path, MonteCarloOptimizer, RandomWalkOptimizer, GeneticOptimizer,
                                      AnnealingOptimizer, HillClimbOptimizer)


def draw_path(path, length, file_name):
    plt.step(path[:, 0], path[:, 1])
    plt.title(f"Length is {length}")
    plt.savefig(file_name)
    plt.clf()


if __name__ == '__main__':
    data = read_tsp()
    data = list(data.itertuples(index=False, name="Point"))
    optimizer = HillClimbOptimizer()
    length, path = find_path(data, optimizer=optimizer)
    draw_path(path, length, f"plot/{type(optimizer).__name__}")
