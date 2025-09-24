.PHONY: init test fmt lint exp

init:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[dev,test,viz]"

test:
	pytest -q

fmt:
	ruff check --fix . || true

lint:
	ruff check .

exp:
	python experiments/run_comparisons.py
