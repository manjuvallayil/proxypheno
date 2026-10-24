import numpy as np

from proxypheno.eval.metrics import mae, pearson, r2, rmse


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert r2(y, y) == 1.0
    assert pearson(y, y) == 1.0


def test_known_errors():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 2.0, 4.0])
    assert rmse(y, pred) == np.sqrt(2 / 3)
    assert mae(y, pred) == 2 / 3
