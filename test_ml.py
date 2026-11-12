import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture(scope="module")
def sample_data():
    """Small, reproducible dataset so the tests run fast in CI."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    y = rng.randint(0, 2, 100)
    return X, y


def test_train_model_uses_random_forest(sample_data):
    """
    Check that train_model returns a RandomForestClassifier, which is the
    expected algorithm for this project.
    """
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_expected_type_and_size(sample_data):
    """
    Check that inference returns a NumPy array with one prediction per
    input row, and that every prediction is either 0 or 1.
    """
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_returns_expected_values():
    """
    Check that compute_model_metrics returns the expected precision, recall,
    and F1 for a small example worked out by hand. There are two actual
    positives and one is predicted correctly, so precision is 1.0, recall
    is 0.5, and F1 is 2/3.
    """
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)
