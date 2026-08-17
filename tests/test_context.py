"""Tests for the context encoder."""

import numpy as np
import pytest

from routernet.context import ContextEncoder


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 6))
    y = rng.integers(0, 3, size=100)
    return X, y


def test_output_shape_and_schema(data):
    X, y = data
    enc = ContextEncoder(n_neighbors=5)
    enc.fit(X, y=y)
    ctx = enc.transform(X)
    assert ctx.shape == (100, 16)
    assert len(enc.FEATURE_NAMES) == 16


def test_fit_transform_matches_transform(data):
    X, y = data
    enc = ContextEncoder(n_neighbors=5)
    ctx1 = enc.fit_transform(X, y=y)
    ctx2 = enc.transform(X)
    assert np.allclose(ctx1, ctx2)


def test_deterministic(data):
    X, y = data
    e1 = ContextEncoder(n_neighbors=5).fit(X, y=y)
    e2 = ContextEncoder(n_neighbors=5).fit(X, y=y)
    assert np.allclose(e1.transform(X), e2.transform(X))


def test_without_labels(data):
    X, _ = data
    enc = ContextEncoder(n_neighbors=5).fit(X)
    ctx = enc.transform(X)
    assert ctx.shape[1] == 16


def test_feature_names_out(data):
    X, y = data
    enc = ContextEncoder().fit(X, y=y)
    names = enc.get_feature_names_out()
    assert list(names) == ContextEncoder.FEATURE_NAMES


def test_knn_neighbor_count(data):
    X, _ = data
    enc = ContextEncoder(n_neighbors=8).fit(X)
    assert enc.knn_model.n_neighbors == 9  # n_neighbors + 1 (self excluded)


def test_generalizes_to_new_data(data):
    X, y = data
    enc = ContextEncoder(n_neighbors=5).fit(X, y=y)
    X_new = np.random.default_rng(8).normal(size=(20, 6))
    ctx = enc.transform(X_new)
    assert ctx.shape == (20, 16)
    assert np.isfinite(ctx).all()
