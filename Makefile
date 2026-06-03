.PHONY: install dev test lint format services-up seed-sample train backtest

PYTHON ?= python3.11
VENV   ?= .venv
PORT   ?= 8004

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[dev]"

dev:
	$(VENV)/bin/uvicorn demandforecast.api.main:app --reload --host 0.0.0.0 --port $(PORT)

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src

format:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

services-up:
	docker compose up -d postgres redis mlflow

seed-sample:
	$(VENV)/bin/python scripts/seed_sample.py

train:
	$(VENV)/bin/python scripts/train.py

backtest:
	$(VENV)/bin/python scripts/backtest.py
