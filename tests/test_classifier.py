"""Tests for RouternetClassifier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from routernet import RouternetClassifier


def make_fast_clf(**kwargs):
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


@pytest.fixture
def dataset():
    X, y = make_classification(
        n_samples=200, n_features=12, n_informative=8, n_classes=3, random_state=42
    )
    return X, y


def test_fit_predict_shape(dataset):
    X, y = dataset
    clf = make_fast_clf()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (200,)
    assert set(np.unique(preds)) <= set(np.unique(y))


def test_classes_and_n_features(dataset):
    X, y = dataset
    clf = make_fast_clf().fit(X, y)
    assert np.array_equal(clf.classes_, np.unique(y))
    assert clf.n_features_in_ == X.shape[1]


def test_predict_proba_sums_to_one(dataset):
    X, y = dataset
    clf = make_fast_clf().fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (200, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_specialist_weights_sum_to_one(dataset):
    X, y = dataset
    clf = make_fast_clf().fit(X, y)
    weights = clf.get_specialist_weights(X)
    assert weights.shape == (200, 3)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(weights >= 0)


def test_context_output(dataset):
    X, y = dataset
    clf = make_fast_clf().fit(X, y)
    ctx = clf.get_context(X)
    assert ctx.shape == (200, 16)


def test_binary_classification():
    X, y = make_classification(
        n_samples=150, n_features=8, n_informative=5, n_classes=2, random_state=1
    )
    clf = make_fast_clf()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_accuracy_reasonable(dataset):
    X, y = dataset
    clf = make_fast_clf().fit(X, y)
    acc = clf.score(X, y)
    # Sanity: must beat random guessing for 3 classes on separable data
    assert acc > 0.6


def test_verbose_flag(dataset, capsys):
    X, y = dataset
    clf = make_fast_clf(verbose=True)
    clf.fit(X, y)
    out = capsys.readouterr().out
    assert "[routernet]" in out


def test_estimator_round_trip_get_set_params(dataset):
    X, y = dataset
    clf = make_fast_clf()
    params = clf.get_params()
    clf2 = RouternetClassifier(**params)
    assert clf2.get_params() == params


def test_clone_works(dataset):
    from sklearn.base import clone

    X, y = dataset
    clf = make_fast_clf()
    clone(clf)


def test_works_in_sklearn_pipeline(dataset):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = dataset
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", make_fast_clf()),
        ]
    )
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (200,)


def test_tiny_class_counts_reduce_oof_folds():
    """StratifiedKFold must not crash even when a class has few samples."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = np.array([0] * 2 + [1] * 18 + [2] * 20)  # class 0 has only 2 samples
    clf = make_fast_clf(oof_folds=5)
    clf.fit(X, y)
    assert clf.predict(X).shape == (40,)


def test_custom_estimator_specialists(dataset):
    from sklearn.tree import DecisionTreeClassifier

    X, y = dataset
    clf = RouternetClassifier(
        n_specialists=3,
        specialist_types=[
            DecisionTreeClassifier(max_depth=5, random_state=0),
            DecisionTreeClassifier(max_depth=7, random_state=1),
            DecisionTreeClassifier(max_depth=9, random_state=2),
        ],
        oof_folds=2,
        gate_hidden=(16, 8),
        gate_epochs=30,
        random_state=42,
    )
    clf.fit(X, y)
    assert clf.predict(X).shape == (200,)
    # User estimators must be cloned, never mutated
    assert clf.specialists_[0] is not clf.specialist_types[0]
