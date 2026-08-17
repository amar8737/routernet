.PHONY: install test lint build publish bench bench-gate clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q --cov=routernet

lint:
	$(PYTHON) -m ruff check src tests

build:
	$(PYTHON) -m build

check:
	twine check dist/*

publish: build
	twine upload dist/*

# Standalone OpenML benchmark (outside the package)
bench:
	$(PYTHON) benchmarks/openml_benchmark.py

bench-gate:
	$(PYTHON) benchmarks/openml_benchmark.py --gate-tasks 6 --limit 10 --n-folds 1

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov src/routernet.egg-info