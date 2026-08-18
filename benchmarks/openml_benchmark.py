"""Standalone OpenML-CC18 benchmark for routernet.

This script is intentionally kept **outside** the ``routernet`` package and is
**not** shipped on PyPI. It imports only the public ``routernet`` API and its
own dependencies (see ``requirements.txt``).

Flow
----
1. Authenticate with OpenML using the ``OPENML_API_KEY`` environment variable
   (or ``--api-key``).
2. Load a curated benchmarking suite (``OpenML-CC18`` by default).
3. **Gate**: evaluate routernet plus baselines on a small subset of tasks
   first. The full run (and any publishing) only proceeds if routernet is
   competitive - by default a win-rate >= ``--gate-win-rate`` against the
   median baseline AND a mean relative accuracy >= ``--gate-min-rel`` of the
   best baseline. Use ``--force`` to skip the gate.
4. **Full run**: evaluate routernet on the requested tasks using OpenML's
   standardized task folds via ``openml.runs.run_model_on_task``.
5. **Publish** (optional, ``--publish``): upload runs to OpenML, only when the
   gate passed.

Example
-------
.. code-block:: bash

    pip install -r benchmarks/requirements.txt
    export OPENML_API_KEY=...
    python benchmarks/openml_benchmark.py --gate-tasks 6 --limit 10 --n-folds 1
    # or, to publish the runs to OpenML:
    python benchmarks/openml_benchmark.py --publish
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

# OpenML's flow serializer resolves the package of each pipeline component from
# ``component.__module__`` and requires ``__version__`` on the top-level module.
# When this script runs directly (``python benchmarks/openml_benchmark.py``) the
# components defined here live in ``__main__``, so expose the version here.
__version__ = "0.1.1"

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from routernet import RouternetClassifier

try:
    import openml  # noqa: F401
    OPENML_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENML_AVAILABLE = False

DEFAULT_SUITE = "OpenML-CC18"
DEFAULT_SPECIALISTS = ["gb", "rf", "et", "svm", "mlp"]


class OpenMLBenchmarkError(RuntimeError):
    """Raised for benchmark-level failures (missing key, bad gate, etc.)."""


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
class _DtypePreprocessor(BaseEstimator, TransformerMixin):
    """Impute/scale numeric columns and one-hot encode categorical columns.

    A plain sklearn transformer (no ``ColumnTransformer`` / ``make_column_selector``)
    so that OpenML's ``sklearn_to_flow`` can serialize it for publishing.
    """

    def __init__(self):
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.numeric_pipe_ = None
        self.categorical_pipe_ = None
        self.n_features_out_ = None

    def fit(self, X, y=None):
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        if hasattr(X, "columns"):
            self.numeric_cols_ = [
                c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
            ]
            self.categorical_cols_ = [
                c for c in X.columns if c not in self.numeric_cols_
            ]
        else:
            self.numeric_cols_ = list(range(X.shape[1]))
            self.categorical_cols_ = []

        self.numeric_pipe_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.categorical_pipe_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                ),
            ]
        )

        if self.numeric_cols_:
            self.numeric_pipe_.fit(self._take(X, self.numeric_cols_))
        if self.categorical_cols_:
            self.categorical_pipe_.fit(self._take(X, self.categorical_cols_))

        n_num = (
            self.numeric_pipe_.transform(self._take(X, self.numeric_cols_)).shape[1]
            if self.numeric_cols_
            else 0
        )
        n_cat = (
            self.categorical_pipe_.transform(
                self._take(X, self.categorical_cols_)
            ).shape[1]
            if self.categorical_cols_
            else 0
        )
        self.n_features_out_ = n_num + n_cat
        return self

    def transform(self, X):
        import numpy as _np

        parts = []
        if self.numeric_cols_:
            parts.append(
                _np.asarray(self.numeric_pipe_.transform(self._take(X, self.numeric_cols_)))
            )
        if self.categorical_cols_:
            cat = self.categorical_pipe_.transform(self._take(X, self.categorical_cols_))
            if hasattr(cat, "toarray"):
                cat = cat.toarray()
            parts.append(_np.asarray(cat))
        if not parts:
            return _np.zeros((len(X), 0))
        return _np.hstack(parts)

    @staticmethod
    def _take(X, cols):
        if hasattr(X, "columns"):
            return X[cols]
        return X[:, cols]

    def get_feature_names_out(self, input_features=None):
        import numpy as _np

        names = []
        if self.numeric_cols_:
            num = self.numeric_pipe_.transform(
                self._take(_np.zeros((1, len(self.numeric_cols_))), self.numeric_cols_)
            )
            names += [f"num_{i}" for i in range(num.shape[1])]
        if self.categorical_cols_:
            names += [f"cat_{i}" for i in range(self.n_features_out_ - len(names))]
        return _np.asarray(names, dtype=object)


def _build_preprocessor() -> _DtypePreprocessor:
    """Build the data preprocessor used by every pipeline (and baselines)."""
    return _DtypePreprocessor()


def _baseline_models() -> dict[str, object]:
    """Create lightweight baseline classifiers (sklearn core + XGBoost if present)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    def wrap(model) -> Pipeline:
        return Pipeline(steps=[("preprocessor", _build_preprocessor()), ("model", model)])

    models: dict[str, object] = {
        "RandomForest": wrap(
            RandomForestClassifier(
                n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
            )
        )
    }
    try:
        import xgboost as xgb

        models["XGBoost"] = wrap(
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
        )
    except ImportError:
        pass
    return models


def _build_pipeline(specialists: list[str], n_specialists: int, seed: int):
    """Build the routernet sklearn pipeline used for OpenML evaluation.

    Handles both numeric and categorical columns (CC18 contains datasets with
    nominal features, e.g. ``kr-vs-kp``).
    """
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            (
                "clf",
                RouternetClassifier(
                    n_specialists=n_specialists,
                    specialist_types=specialists[:n_specialists],
                    oof_folds=3,
                    use_global_fallback=True,
                    verbose=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def _task_detailed(
    task,
    routernet_model,
    specialist_names: list[str],
    n_folds: int,
) -> tuple[dict[str, float], int]:
    """Evaluate routernet, its specialists, uniform ensemble, and baselines.

    OpenML 0.15 removed the ``n_folds`` argument from
    :func:`openml.runs.run_model_on_task`, so we evaluate the requested number
    of folds manually using the task's official train/test splits.

    Specialists and the uniform ensemble are extracted from the *already-fitted*
    routernet pipeline, making this a routing ablation over identical experts:
    any ``Uniform`` vs ``Routernet`` difference is attributable to routing alone.

    External baselines are built fresh per task: refitting the same estimator
    instances across tasks with different class counts corrupts internal state
    (e.g. XGBoost keeps ``objective='multi:softprob'`` after a multiclass fit
    and then fails on a binary refit).
    """
    import sklearn.metrics
    from sklearn.preprocessing import LabelEncoder

    baselines = _baseline_models()

    dataset = task.get_dataset()
    df = dataset.get_data(target=task.target_name)[0]
    y = dataset.get_data(target=task.target_name)[1].to_numpy()

    if y.dtype.kind not in "iufb":
        y = LabelEncoder().fit_transform(y)

    n_repeats, n_folds_max, n_samples = task.get_split_dimensions()
    n_folds = min(n_folds, n_folds_max) if n_folds > 0 else n_folds_max

    preprocessor = routernet_model.named_steps["preprocessor"]
    clf = routernet_model.named_steps["clf"]

    model_names = ["Routernet", "Uniform", *specialist_names, *baselines.keys()]
    scores: dict[str, list[float]] = {name: [] for name in model_names}

    for fold in range(n_folds):
        train_idx, test_idx = task.get_train_test_split_indices(
            fold=fold, repeat=0, sample=0
        )
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Routernet pipeline
        routernet_model.fit(X_train, y_train)
        pred = routernet_model.predict(X_test)
        scores["Routernet"].append(sklearn.metrics.accuracy_score(y_test, pred))

        # Specialists + uniform ensemble (routing ablation)
        X_test_proc = preprocessor.transform(X_test)
        spec_probas: list[np.ndarray] = []
        for i, specialist in enumerate(clf.specialists_):
            proba = clf._get_proba(specialist, X_test_proc)
            spec_probas.append(proba)
            scores[specialist_names[i]].append(
                sklearn.metrics.accuracy_score(y_test, np.argmax(proba, axis=1))
            )
        if spec_probas:
            uniform = np.mean(spec_probas, axis=0)
            scores["Uniform"].append(
                sklearn.metrics.accuracy_score(y_test, np.argmax(uniform, axis=1))
            )

        # External baselines
        for name, baseline in baselines.items():
            baseline.fit(X_train, y_train)
            scores[name].append(
                sklearn.metrics.accuracy_score(y_test, baseline.predict(X_test))
            )

    return (
        {name: float(np.mean(v)) for name, v in scores.items() if v},
        int(n_folds),
    )


def _gate_passed(
    gate_results: dict[str, dict[str, float]],
    win_rate: float,
    min_rel: float,
    baseline_names: list[str],
):
    """Decide whether routernet is competitive enough to proceed.

    The gate compares routernet only against the independent external
    baselines (RF/XGB). Specialists and the uniform ensemble are routing
    ablations derived from routernet itself, so they never participate in the
    gate decision.
    """
    if not gate_results:
        return False, {"reason": "no gate results"}

    routernet_col = "Routernet"
    baselines = [b for b in baseline_names if any(b in row for row in gate_results.values())]
    if not baselines:
        return False, {"reason": "no baseline scores available for the gate"}

    per_task = []
    for row in gate_results.values():
        if routernet_col not in row:
            continue
        base_accs = [row[m] for m in baselines if m in row]
        if not base_accs:
            continue
        per_task.append(
            {
                "routernet": row[routernet_col],
                "best_baseline": max(base_accs),
                "median_baseline": float(np.median(base_accs)),
                "best_baseline_name": baselines[np.argmax(base_accs)],
            }
        )

    if not per_task:
        return False, {"reason": "routernet produced no gate scores"}

    wins = sum(1 for r in per_task if r["routernet"] >= r["median_baseline"])
    achieved_win_rate = wins / len(per_task)
    rel_acc = float(np.mean([r["routernet"] / r["best_baseline"] for r in per_task]))
    passed = achieved_win_rate >= win_rate and rel_acc >= min_rel

    summary = {
        "n_tasks": len(per_task),
        "win_rate": achieved_win_rate,
        "required_win_rate": win_rate,
        "rel_accuracy": rel_acc,
        "required_rel_accuracy": min_rel,
        "passed": passed,
        "per_task": per_task,
    }
    return passed, summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _save_results(results: list[dict], out_dir: Path) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = out_dir / "openml_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(results)} results to {csv_path}")


def _save_markdown(results: list[dict], out_dir: Path, specialist_names: list[str]) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    pivot = df.pivot_table(
        index="dataset", columns="model", values="accuracy", aggfunc="mean"
    ).round(4)

    order = ["Routernet", *specialist_names, "Uniform", "RandomForest", "XGBoost"]
    order = [c for c in order if c in pivot.columns]
    order += [c for c in pivot.columns if c not in order]
    pivot = pivot.reindex(columns=order)

    md_path = out_dir / "openml_results.md"
    md_path.write_text(
        "# Routernet OpenML Benchmark\n\n" + pivot.to_markdown() + "\n",
        encoding="utf-8",
    )
    print(f"Saved markdown report to {md_path}")


def _save_summary(results: list[dict], out_dir: Path, specialist_names: list[str]) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    pivot = df.pivot_table(
        index="dataset", columns="model", values="accuracy", aggfunc="mean"
    )

    summary = pd.DataFrame(index=pivot.index)
    spec_cols = [c for c in specialist_names if c in pivot.columns]
    summary["BestExpert"] = pivot[spec_cols].max(axis=1) if spec_cols else np.nan
    if "Uniform" in pivot.columns:
        summary["Uniform"] = pivot["Uniform"]
    summary["RouterNet"] = pivot["Routernet"]
    if "Uniform" in pivot.columns:
        summary["RouterNet_vs_Uniform"] = summary["RouterNet"] - summary["Uniform"]
    for col in ("RandomForest", "XGBoost"):
        if col in pivot.columns:
            summary[col] = pivot[col]
    summary = summary.round(4)

    md_path = out_dir / "openml_summary.md"
    md_path.write_text(
        "# Routernet OpenML Benchmark Summary\n\n"
        "> `RouterNet_vs_Uniform` is the routing ablation: > 0 means the router "
        "adds value over the exact experts it routes; ~0 means routing is "
        "unnecessary; < 0 means routing is actively hurting.\n\n"
        + summary.to_markdown()
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved summary report to {md_path}")


def _save_gate_summary(gate_summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gate_summary.json"
    path.write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")
    print(f"Saved gate summary to {path}")


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def _load_suite(suite_name: str) -> list:
    """Load tasks from the suite."""
    import openml

    suite = openml.study.get_suite(suite_name)
    task_ids = list(suite.tasks)
    print(f"Loaded suite '{suite_name}' with {len(task_ids)} task(s).")
    return [openml.tasks.get_task(tid) for tid in task_ids]


def _filter_tasks_by_size(tasks, min_samples=None, max_samples=None) -> list:
    """Filter tasks by dataset size (requires dataset download)."""
    if min_samples is None and max_samples is None:
        return tasks
    kept = []
    for task in tasks:
        dataset = task.get_dataset()
        X, _, _, _ = dataset.get_data()
        n = np.asarray(X).shape[0]
        if min_samples and n < min_samples:
            print(f"  skipping {dataset.name} ({n} samples < {min_samples})")
            continue
        if max_samples and n > max_samples:
            print(f"  skipping {dataset.name} ({n} samples > {max_samples})")
            continue
        kept.append(task)
    return kept


def run_benchmark(args: argparse.Namespace) -> int:
    """Execute the gate + full benchmark. Returns process exit code."""
    if not OPENML_AVAILABLE:
        raise OpenMLBenchmarkError(
            "openml is not installed. Install with: pip install -r benchmarks/requirements.txt"
        )
    import openml

    api_key = args.api_key or os.environ.get("OPENML_API_KEY")
    if not api_key:
        raise OpenMLBenchmarkError(
            "No OpenML API key found. Set the OPENML_API_KEY environment variable "
            "or pass --api-key."
        )
    openml.config.apikey = api_key

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ROUTERNET OPENML BENCHMARK")
    print("=" * 70)

    tasks = _load_suite(args.suite)
    tasks = _filter_tasks_by_size(tasks, args.min_samples, args.max_samples)

    specialists = (
        [s.strip() for s in args.specialists.split(",") if s.strip()]
        if args.specialists
        else DEFAULT_SPECIALISTS
    )
    specialist_names = [s.upper() for s in specialists[: args.n_specialists]]
    baselines = _baseline_models()
    routernet_model = _build_pipeline(specialists, args.n_specialists, args.seed)

    # -- Gate -------------------------------------------------------------- #
    gate_results = None
    passed = True
    gate_summary = None
    if not args.force:
        print(
            f"\n[GATE] Evaluating on {args.gate_tasks} task(s), "
            f"{args.gate_folds} fold(s) each..."
        )
        gate_tasks = tasks[: args.gate_tasks]
        gate_results = {}
        for task in gate_tasks:
            tid = task.task_id
            name = task.get_dataset().name
            print(f"  - task {tid} ({name}):")
            try:
                accs, nf = _task_detailed(
                    task, routernet_model, specialist_names, args.gate_folds
                )
                gate_results[tid] = accs
                for model_name, acc in accs.items():
                    print(f"      {model_name:<14} acc={acc:.4f} (folds={nf})")
            except Exception as exc:  # noqa: BLE001
                print(f"      FAILED: {exc}")

        passed, gate_summary = _gate_passed(
            gate_results, args.gate_win_rate, args.gate_min_rel, list(baselines.keys())
        )

        print(
            f"\nGate: win_rate={gate_summary['win_rate']:.3f} "
            f"(req {gate_summary['required_win_rate']}), "
            f"rel_acc={gate_summary['rel_accuracy']:.3f} "
            f"(req {gate_summary['required_rel_accuracy']})"
        )
        if not passed:
            print(
                "Routernet did NOT pass the performance gate. "
                "Skipping the full benchmark and any publishing. "
                "Use --force to override."
            )
            _save_gate_summary(gate_summary, out_dir)
            return 1

        print("Gate PASSED. Proceeding with the full benchmark.")
        _save_gate_summary(gate_summary, out_dir)

    # -- Full run ---------------------------------------------------------- #
    full_tasks = tasks
    if args.limit:
        full_tasks = tasks[: args.limit]

    print(
        f"\n[FULL] Running routernet on {len(full_tasks)} task(s), "
        f"{args.n_folds} fold(s) each."
    )
    all_results: list[dict] = []

    if gate_results:
        for tid, row in gate_results.items():
            task = next(t for t in full_tasks if t.task_id == tid)
            dataset_name = task.get_dataset().name
            for model_name, acc in row.items():
                all_results.append(
                    {
                        "task_id": tid,
                        "dataset": dataset_name,
                        "model": model_name,
                        "accuracy": acc,
                        "folds": args.gate_folds,
                        "gate": True,
                    }
                )

    done_ids = {r["task_id"] for r in all_results if r.get("gate")}
    for task in full_tasks:
        tid = task.task_id
        if tid in done_ids:
            continue
        dataset_name = task.get_dataset().name
        print(f"  - task {tid} ({dataset_name})...", end=" ", flush=True)
        try:
            accs, nf = _task_detailed(
                task, routernet_model, specialist_names, args.n_folds
            )
            for model_name, acc in accs.items():
                all_results.append(
                    {
                        "task_id": tid,
                        "dataset": dataset_name,
                        "model": model_name,
                        "accuracy": acc,
                        "folds": nf,
                        "gate": False,
                    }
                )
            print(f"acc={accs['Routernet']:.4f} (folds={nf})")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            all_results.append(
                {
                    "task_id": tid,
                    "dataset": dataset_name,
                    "model": "Routernet",
                    "accuracy": np.nan,
                    "folds": 0,
                    "error": str(exc),
                    "gate": False,
                }
            )

    # -- Publish ----------------------------------------------------------- #
    if args.publish and passed:
        print("\n[PUBLISH] Uploading routernet runs to OpenML...")
        published = 0
        failed = []
        for task in full_tasks:
            try:
                run = openml.runs.run_model_on_task(
                    routernet_model, task, seed=1
                )
                run.publish()
                published += 1
                print(f"  uploaded run {run.run_id} for task {task.task_id}")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  publish failed for task {task.task_id}:\n"
                    f"{traceback.format_exc()}"
                )
                failed.append({"task_id": task.task_id, "error": str(exc)})
        print(f"Published {published} run(s).")
        (out_dir / "published_runs.json").write_text(
            json.dumps({"published": published, "failed": failed}),
            encoding="utf-8",
        )
    elif args.publish:
        print("Skipping publish: gate did not pass.")

    _save_results(all_results, out_dir)
    _save_markdown(all_results, out_dir, specialist_names)
    _save_summary(all_results, out_dir, specialist_names)

    accs = [
        r["accuracy"]
        for r in all_results
        if r.get("model") == "Routernet"
        and not np.isnan(r.get("accuracy", np.nan))
    ]
    if accs:
        print(
            f"\nMean routernet accuracy across {len(accs)} task(s): {np.mean(accs):.4f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="openml_benchmark",
        description="Benchmark routernet on OpenML-CC18 with a performance gate.",
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="OpenML study suite name")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenML API key (defaults to OPENML_API_KEY env var)",
    )
    parser.add_argument(
        "--gate-tasks",
        type=int,
        default=6,
        help="Number of tasks to use for the performance gate (default: 6)",
    )
    parser.add_argument(
        "--gate-folds",
        type=int,
        default=1,
        help="Number of OpenML folds to evaluate in the gate (default: 1)",
    )
    parser.add_argument(
        "--gate-win-rate",
        type=float,
        default=0.5,
        help="Required win rate vs median baseline for the gate (default: 0.5)",
    )
    parser.add_argument(
        "--gate-min-rel",
        type=float,
        default=0.97,
        help="Required mean relative accuracy vs best baseline (default: 0.97)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of OpenML folds for the full run (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of tasks in the full run (default: 10, partial run)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Skip datasets with fewer samples than this",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Skip datasets with more samples than this",
    )
    parser.add_argument(
        "--specialists",
        default=None,
        help="Comma-separated specialist types (e.g. 'gb,rf,et,svm,mlp')",
    )
    parser.add_argument(
        "--n-specialists",
        type=int,
        default=5,
        help="Number of specialist models (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the performance gate entirely",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload runs to OpenML (only honored if the gate passes)",
    )
    parser.add_argument("--output", default="results", help="Output directory (default: results)")
    args = parser.parse_args(argv)

    try:
        return run_benchmark(args)
    except OpenMLBenchmarkError as exc:
        print(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
