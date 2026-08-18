# routernet

Per-sample specialist routing ensemble with learned context awareness.

`routernet` is a scikit-learn compatible ensemble algorithm. It trains a pool
of diverse specialist models and a learned **gating network** that routes each
sample to the specialists best suited for it, based on a learned 16-dimensional
**context embedding** (neighborhood geometry, feature statistics, label
structure and specialist agreement).

## Installation

```bash
pip install routernet
```

Core dependencies are kept minimal: `numpy`, `scipy`, `pandas`,
`scikit-learn`. Install the optional boosting specialists (`cat`/`xgb`) with:

```bash
pip install routernet[boost]      # xgboost, lightgbm, catboost
```

## Quick start

```python
from routernet import RouternetClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=20, n_informative=12,
                           n_classes=3, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RouternetClassifier(n_specialists=3, random_state=42)
clf.fit(X_tr, y_tr)
print(f"Accuracy: {accuracy_score(y_te, clf.predict(X_te)):.3f}")
```

Regression works the same way with `RouternetRegressor`.

## How it works

1. **Specialists** — a pool of diverse models (`gb`, `rf`, `svm`, `mlp`, `dt`,
   `et`, `cat`, `xgb`, or any scikit-learn estimator) are trained. Out-of-fold
   (OOF) predictions avoid leakage when training the gate.
2. **Context encoder** — each sample is embedded into 16 features describing
   its local neighborhood, feature statistics and specialist agreement.
3. **Gating network** — a small MLP (trained with the built-in autograd engine)
   maps context embeddings to per-sample specialist weights by directly
   optimizing the ensemble cross-entropy loss.
4. **Confidence gate** — a learned per-sample fallback blends in the uniform
   ensemble when routing would be unreliable.

Because everything is a `BaseEstimator`, `RouternetClassifier` /
`RouternetRegressor` work in `Pipeline`s, `GridSearchCV`, and OpenML's
`run_model_on_task`.

## Interpretability

```python
weights = clf.get_specialist_weights(X)   # per-sample specialist weights
context = clf.get_context(X)              # 16-dim context embeddings
analysis = analyze_routing(clf, X, y)     # routing summary
print_routing_analysis(analysis)
```

## Development

```bash
pip install -e ".[dev]"
make test       # pytest + coverage
make lint       # ruff
make build      # sdist + wheel
make check      # twine check
```

## Benchmarking on OpenML

The OpenML-CC18 benchmark lives **outside** this package in [`benchmarks/`](benchmarks/)
and is **not** shipped on PyPI. It runs a performance gate before the full
run and publishes to OpenML only if routernet performs well:

```bash
pip install -r benchmarks/requirements.txt
export OPENML_API_KEY=YOUR_OPENML_API_KEY
python benchmarks/openml_benchmark.py --gate-tasks 6 --limit 10
```

See [`benchmarks/README.md`](benchmarks/README.md) for details.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).