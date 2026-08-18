#!/usr/bin/env bash
# Run the routernet OpenML-CC18 benchmark locally.
# Usage: ./run_benchmark.sh [--publish]
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "${OPENML_API_KEY:-}" ]]; then
  echo "ERROR: OPENML_API_KEY not set. Add it to .env or export it." >&2
  exit 1
fi

ARGS=()
if [[ "${1:-}" == "--publish" ]]; then
  ARGS+=(--publish)
fi

exec python -u benchmarks/openml_benchmark.py \
  --gate-tasks 6 \
  --limit 10 \
  --n-folds 1 \
  --output results \
  "${ARGS[@]}"
