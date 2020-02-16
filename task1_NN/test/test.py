from sklearn.metrics import accuracy_score, precision_score, recall_score

from task1_NN.main import get_precision_recall_accuracy

class TestMetrics:
    def test_all_the_same(self):
        pred = [0] * 5
        true = [0] * 5
        (precision, precision2), (recall, recall2), accuracy = get_precision_recall_accuracy(pred, true)
        assert precision == precision_score(true, pred)
        assert recall == recall_score(true, pred)
        assert accuracy == accuracy_score(true, pred)
        accuracy2 = accuracy
        pred = [1] * 5
        true = [1] * 5
        (precision, _), (recall, _), accuracy = get_precision_recall_accuracy(pred, true)
        assert precision == precision_score(true, pred)
        assert recall == recall_score(true, pred)
        assert accuracy == accuracy_score(true, pred)
        assert precision == precision2
        assert recall == recall2
        assert accuracy == accuracy2


    def test_some_data(self):
        pred = [0, 0, 1, 1, 0]
        true = [0, 1, 0, 1, 0]
        (precision, precision2), (recall, recall2), accuracy = get_precision_recall_accuracy(pred, true)
        assert precision == precision_score(true, pred)
        assert recall == recall_score(true, pred)
        assert accuracy == accuracy_score(true, pred)
        accuracy2 = accuracy
        pred = [0 if p else 1 for p in pred]
        true = [0 if t else 1 for t in true]
        (precision, _), (recall, _), accuracy = get_precision_recall_accuracy(pred, true)
        assert precision == precision_score(true, pred)
        assert recall == recall_score(true, pred)
        assert accuracy == accuracy_score(true, pred)
        assert precision == precision2
        assert recall == recall2
        assert accuracy == accuracy2
