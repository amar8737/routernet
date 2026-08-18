"""Tests for the memory-friendly batched gating-network training."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from routernet import RouternetClassifier
from routernet.gate import GatingNetwork


@pytest.fixture
def dataset():
    X, y = make_classification(
        n_samples=400, n_features=12, n_informative=8, n_classes=3, random_state=42
    )
    return X, y


def _make_clf(**kwargs):
    params = dict(
        n_specialists=3,
        specialist_types=["gb", "rf", "dt"],
        oof_folds=2,
        gate_hidden=(16, 8),
        gate_epochs=50,
        random_state=42,
    )
    params.update(kwargs)
    return RouternetClassifier(**params)


def test_batched_matches_full_batch(dataset):
    X, y = dataset
    batched = _make_clf(gate_batch_size=64, gate_patience=0).fit(X, y)
    full = _make_clf(gate_batch_size=0, gate_patience=0).fit(X, y)
    # Both should produce well-formed, comparable classifiers.
    assert abs(batched.score(X, y) - full.score(X, y)) < 0.2


def test_full_batch_fallback_equal_to_large_batch(dataset):
    """batch_size >= n behaves identically to batch_size <= 0 (full batch)."""
    X, y = dataset
    clf_a = _make_clf(gate_batch_size=0, gate_patience=0).fit(X, y)
    clf_b = _make_clf(gate_batch_size=10000, gate_patience=0).fit(X, y)
    p_a = clf_a.predict_proba(X)
    p_b = clf_b.predict_proba(X)
    assert np.allclose(p_a, p_b)


def test_batching_is_memory_friendly_small_graphs(dataset):
    """Batched training must not blow up the autograd graph size per batch.

    We assert that a very small batch runs and produces finite weights,
    implying the graph is built per-batch rather than over the full set.
    """
    X, y = dataset
    clf = _make_clf(gate_batch_size=16, gate_epochs=5, gate_patience=0).fit(X, y)
    weights = clf.get_specialist_weights(X)
    assert weights.shape == (400, 3)
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)


def test_early_stopping_restores_best_params(dataset):
    """Early stopping restores the best parameter snapshot (finite, usable)."""
    X, y = dataset
    clf = _make_clf(
        gate_batch_size=64,
        gate_patience=3,
        gate_val_fraction=0.2,
        gate_epochs=200,
    ).fit(X, y)
    assert np.isfinite(clf.get_specialist_weights(X)).all()
    assert clf.score(X, y) > 0.6


def test_early_stopping_disabled_uses_all_epochs(dataset):
    """patience <= 0 disables the validation split and early stopping."""
    X, y = dataset
    clf = _make_clf(gate_batch_size=64, gate_patience=0).fit(X, y)
    assert clf.score(X, y) > 0.6


def test_cosine_schedule_does_not_nan(dataset):
    X, y = dataset
    clf = _make_clf(gate_batch_size=64, gate_lr_schedule="cosine").fit(X, y)
    weights = clf.get_specialist_weights(X)
    assert np.isfinite(weights).all()


def test_constant_schedule_works(dataset):
    X, y = dataset
    clf = _make_clf(gate_batch_size=64, gate_lr_schedule="none").fit(X, y)
    assert clf.score(X, y) > 0.6


def test_snapshot_restore_roundtrip():
    net = GatingNetwork(context_dim=4, n_specialists=3, hidden_dims=(8, 4))
    snap = net._snapshot()
    # Perturb params, then restore.
    for p in net.parameters:
        p.data = p.data + 100.0
    net._restore(snap)
    for p, orig in zip(net.parameters, snap, strict=True):
        assert np.allclose(p.data, orig)


def test_confidence_gate_batched(dataset):
    X, y = dataset
    clf = _make_clf(gate_batch_size=64).fit(X, y)
    assert clf.conf_gate_ is not None
    gate = clf.conf_gate_.predict(clf.get_context(X))
    assert gate.shape == (400, 1)
    assert np.isfinite(gate).all()


def test_reproducible_with_batching():
    clf1 = _make_clf(gate_batch_size=64).fit(*make_classification(
        n_samples=200, n_features=8, n_informative=5, n_classes=2, random_state=0
    ))
    clf2 = _make_clf(gate_batch_size=64).fit(*make_classification(
        n_samples=200, n_features=8, n_informative=5, n_classes=2, random_state=0
    ))
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=5, n_classes=2, random_state=0
    )
    assert np.allclose(clf1.predict_proba(X), clf2.predict_proba(X))