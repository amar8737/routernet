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
python benchmarks/openml_benchmark.py --gate-tasks 6 --limit 20 --pools mixed,trees,nn
```

See [`benchmarks/README.md`](benchmarks/README.md) for details, and
[`THEORY.md`](THEORY.md) for a rigorous account of *when* sample-conditional
routing is provably better than a fixed-weight ensemble.

## Benchmark results

Evaluated on the first 10 tasks of [OpenML-CC18](https://www.openml.org/s/99),
using OpenML's official train/test splits (1 fold each), with a 6-task
performance gate (win-rate vs. median baseline and relative accuracy vs. best
baseline).

| dataset             | BestExpert | Uniform | RouterNet | Δ vs Uniform | RandomForest | XGBoost |
|:--------------------|-----------:|--------:|----------:|-------------:|-------------:|--------:|
| balance-scale       |      1.000 |  0.9206 |    0.9524 |       +0.032 |       0.8889 |  0.9365 |
| breast-w            |      0.9857 |  0.9571 |    0.9571 |       0.000  |       0.9714 |  0.9714 |
| cmc                 |      0.5946 |  0.5473 |    0.5676 |       +0.020 |       0.5743 |  0.5743 |
| kr-vs-kp            |      0.9938 |  0.9906 |    0.9969 |       +0.006 |       0.9875 |  0.9938 |
| letter              |      0.9725 |  0.9745 |    0.9740 |       -0.001 |       0.8685 |  0.9550 |
| mfeat-factors       |      0.9750 |  0.9800 |    0.9800 |       0.000  |       0.9600 |  0.9600 |
| mfeat-fourier       |      0.8500 |  0.8300 |    0.8250 |       -0.005 |       0.8250 |  0.8500 |
| mfeat-karhunen      |      0.9800 |  0.9850 |    0.9800 |       -0.005 |       0.9700 |  0.9550 |
| mfeat-morphological |      0.7000 |  0.6650 |    0.6650 |       0.000  |       0.6700 |  0.6700 |
| mfeat-zernike       |      0.8100 |  0.7800 |    0.7950 |       +0.015 |       0.7400 |  0.7950 |

`Δ vs Uniform` is the routing ablation: > 0 means the router adds value over the
exact experts it routes; ~0 means routing is unnecessary. Routernet matches or
improves on the uniform ensemble in 7 of 10 tasks.

The full per-dataset × model matrix and raw CSVs are in [`results/`](results/);
see [`benchmarks/README.md`](benchmarks/README.md) to reproduce.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).