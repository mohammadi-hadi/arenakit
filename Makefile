.PHONY: install test lint results examples all clean

PY ?= .venv/bin/python

install:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests examples
	$(PY) -m ruff format --check src tests examples
	$(PY) -m mypy src

# Regenerate the committed demo results. CI runs this and fails on any diff,
# so the numbers in the README cannot drift away from the code.
results:
	$(PY) -m arenakit.cli demo --out results

# The case studies download real votes, so they are not part of CI.
examples:
	$(PY) examples/lmarena/audit_lmarena.py
	$(PY) examples/mtbench/audit_mtbench.py

all: lint test results

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
