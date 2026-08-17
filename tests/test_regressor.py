"""Tests for RouternetRegressor."""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from routernet import RouternetRegressor


@pytest.fixture
def dataset():
    X, y = make_regression(
        n_samples=150, n_features=8, n_informative=6, noise=0.5, random_state=42
    )
    return X, y


def test_fit_predict(dataset):
    X, y = dataset
    reg = RouternetRegressor(
        n_specialists=3,
        specialist_types=["gb", "rf", "mlp"],
        random_state=42,
    )
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (150,)
    assert np.all(np.isfinite(preds))


def test_prediction_correlation(dataset):
    X, y = dataset
    reg = RouternetRegressor(
        n_specialists=3,
        specialist_types=["rf", "rf", "rf"],
        random_state=42,
    )
    reg.fit(X, y)
    preds = reg.predict(X)
    corr = np.corrcoef(y, preds)[0, 1]
    assert corr > 0.8


def test_specialist_weights_shape(dataset):
    X, y = dataset
    reg = RouternetRegressor(
        n_specialists=2, specialist_types=["rf", "gb"], random_state=42
    )
    reg.fit(X, y)
    w = reg.get_specialist_weights(X)
    assert w.shape == (150, 2)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)


def test_context_shape(dataset):
    X, y = dataset
    reg = RouternetRegressor(n_specialists=2, specialist_types=["rf", "gb"], random_state=42)
    reg.fit(X, y)
    assert reg.get_context(X).shape == (150, 16)


def test_works_in_pipeline(dataset):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = dataset
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "reg",
                RouternetRegressor(
                    n_specialists=2, specialist_types=["rf", "gb"], random_state=42
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (150,)
