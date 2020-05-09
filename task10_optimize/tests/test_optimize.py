import pytest
import numpy as np

from task10_optimize.optimize import l1_path_len


@pytest.mark.parametrize("path, expected", [(np.array([[565, 575], [ 25, 185], [345, 750], [945, 685]]), 2480)])
def test_l1_path_len(path, expected):
    assert expected == l1_path_len(path)
