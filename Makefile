.PHONY: install dev test lint format services-up services-down seed-sample train backtest \
        research-download research-preprocess research-train research-eval notebook mlflow-ui

PYTHON   ?= python3.11
VENV     ?= .venv
PORT     ?= 8004
CONFIG   ?= configs/lgbm_recursive.yaml
RUN_ID   ?= latest

# --- Service (FastAPI) ---
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

services-down:
	docker compose down

seed-sample:
	$(VENV)/bin/python scripts/seed_sample.py

train:
	$(VENV)/bin/python scripts/train.py

backtest:
	$(VENV)/bin/python scripts/backtest.py

# --- ML research (M5 benchmark, hợp nhất từ nhánh kaggle) ---
research-download:
	PYTHONPATH=$(shell pwd) python src/data/download.py

research-preprocess:
	PYTHONPATH=$(shell pwd) python src/data/preprocess.py
	PYTHONPATH=$(shell pwd) python src/data/features.py

research-train:
	PYTHONPATH=$(shell pwd) python src/train.py --config $(CONFIG)

research-eval:
	PYTHONPATH=$(shell pwd) python src/train.py --config $(CONFIG) --eval-only --run-id $(RUN_ID)

notebook:
	jupyter lab notebooks/

mlflow-ui:
	mlflow ui --port 5000
