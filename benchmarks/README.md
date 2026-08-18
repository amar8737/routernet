# OpenML Benchmark (standalone)

This directory contains the **standalone** OpenML benchmark for routernet.

It is deliberately kept **outside** the `routernet` package:

- It is **not** shipped in the PyPI sdist/wheel (`pip install routernet` never
  includes or requires it).
- It imports only the **public** `routernet` API (`RouternetClassifier`).
- It has its **own** dependency list (see `requirements.txt`).

## Setup

```bash
pip install -r benchmarks/requirements.txt   # installs openml + friends
pip install -e .                             # installs the routernet package
export OPENML_API_KEY=YOUR_OPENML_API_KEY
```

## Usage

Run the **gate-then-benchmark** flow (partial by default — 6 gate tasks, then
up to 10 tasks in the full run, 1 fold each):

```bash
python benchmarks/openml_benchmark.py
```

The **gate** first checks whether routernet is competitive against baselines
(RandomForest, and XGBoost if installed) on the smallest tasks. The full run
**and** any publishing only proceed if routernet wins at least
`--gate-win-rate` (default 0.5) against the median baseline and reaches at
least `--gate-min-rel` (default 0.97) of the best baseline's accuracy.

Skip the gate entirely with `--force`. Upload runs to OpenML only with
`--publish` (honored only when the gate passes).

### Options

```
--suite           OpenML study suite (default: OpenML-CC18)
--api-key         API key (defaults to OPENML_API_KEY env var)
--gate-tasks      tasks used for the gate           (default: 6)
--gate-folds      folds used for the gate           (default: 1)
--gate-win-rate   required win-rate vs median base  (default: 0.5)
--gate-min-rel    required rel-accuracy vs best base(default: 0.97)
--n-folds         folds for the full run            (default: 1)
--pools           specialist pools to benchmark      (default: mixed; options:
                  mixed, trees, nn, or comma-separated e.g. mixed,nn)
--specialists     comma-separated specialist types (overrides --pools)
--limit           max tasks in the full run         (default: 10)
--min-samples     skip datasets smaller than this
--max-samples     skip datasets larger than this
--specialists     comma-separated specialist types  (default: gb,rf,et,svm,mlp)
--n-specialists   number of specialists             (default: 5)
--seed            random seed                       (default: 42)
--force           skip the performance gate
--publish         upload runs to OpenML (only if gate passes)
--output          output directory                  (default: results)
```

### Examples

```bash
# Partial benchmark (gate + 10 tasks, 1 fold each)
python benchmarks/openml_benchmark.py

# Benchmark 20 tasks across all three specialist pools (mixed/trees/nn)
python benchmarks/openml_benchmark.py --limit 20 --pools mixed,trees,nn

# Full CC18 suite, 3 folds per task, no upload
python benchmarks/openml_benchmark.py --limit 0 --n-folds 3 --max-samples 20000

# Skip the gate and publish everything (use with care)
python benchmarks/openml_benchmark.py --force --publish
```

## Output

Results are written to `results/`:

- `gate_summary.json` — gate outcome (win-rate, relative accuracy, per-task detail)
- `openml_results.csv`  — per-task routernet accuracy
- `openml_results.md`   — markdown report
