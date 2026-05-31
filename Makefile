PYTHON   := PYTHONPATH=$(shell pwd) python
CONFIG   ?= configs/lgbm_recursive.yaml
RUN_ID   ?= latest

.PHONY: download preprocess train eval notebook mlflow-ui test

download:
	$(PYTHON) src/data/download.py

preprocess:
	$(PYTHON) src/data/preprocess.py
	$(PYTHON) src/data/features.py

train:
	$(PYTHON) src/train.py --config $(CONFIG)

eval:
	$(PYTHON) src/train.py --config $(CONFIG) --eval-only --run-id $(RUN_ID)

notebook:
	jupyter lab notebooks/

mlflow-ui:
	mlflow ui --port 5000

test:
	python -m pytest tests/ -v
