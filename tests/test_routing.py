"""Tests for routing analysis utilities and reproducibility."""

import numpy as np
from sklearn.datasets import make_classification

from routernet import RouternetClassifier, analyze_routing, print_routing_analysis


def _fit_fast_clf(seed=42):
    X, y = make_classification(
        n_samples=120, n_features=8, n_informative=5, n_classes=3, random_state=0
    )
    clf = RouternetClassifier(
        n_specialists=3,
        specialist_types=["gb", "rf", "dt"],
        oof_folds=2,
        gate_hidden=(16, 8),
        gate_epochs=30,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf, X, y


def test_analyze_routing():
    clf, X, y = _fit_fast_clf()
    analysis = analyze_routing(clf, X, y, sample_indices=np.arange(5))
    assert "weights" in analysis
    assert "context" in analysis
    assert "primary_specialist" in analysis
    assert "weight_entropy" in analysis
    assert analysis["weights"].shape == (5, 3)
    assert analysis["primary_specialist"].shape == (5,)


def test_print_routing_analysis(capsys):
    clf, X, y = _fit_fast_clf()
    analysis = analyze_routing(clf, X, y)
    print_routing_analysis(analysis)
    out = capsys.readouterr().out
    assert "ROUTERNET ROUTING ANALYSIS" in out


def test_reproducible_with_same_seed():
    clf1, X, y = _fit_fast_clf(seed=42)
    clf2, X, y = _fit_fast_clf(seed=42)
    p1 = clf1.predict_proba(X)
    p2 = clf2.predict_proba(X)
    assert np.allclose(p1, p2)


def test_different_seeds_differ():
    clf1, X, y = _fit_fast_clf(seed=42)
    clf2, X, y = _fit_fast_clf(seed=1)
    p1 = clf1.predict_proba(X)
    p2 = clf2.predict_proba(X)
    assert not np.allclose(p1, p2)
