import pytest

from task3_tree.criterion import gini, entropy


class TestGini:
    def test_eq(self):
        assert 0.5 == gini([1, 0, 1, 0])

    @pytest.mark.parametrize("data", [[1, 1], [0, 0]])
    def test_all_the_same(self, data):
        assert 0.0 == gini(data)


class TestEntropy:
    def test_eq(self):
        assert 1 == entropy([1, 0, 1, 0])

    @pytest.mark.parametrize("data", [[1, 1], [0, 0]])
    def test_all_the_same(self, data):
        assert 0.0 == entropy(data)
