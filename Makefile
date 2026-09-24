PYTHON ?= python
DEMO_MODELS = $(CURDIR)/artifacts/demo/models
DEMO_LOG = $(CURDIR)/artifacts/demo/prediction_logs.jsonl

.PHONY: demo api-demo ui-demo test lint features train

demo:
	$(PYTHON) -m scripts.run_demo

api-demo:
	CREDIT_RISK_MODEL_DIR=$(DEMO_MODELS) CREDIT_RISK_LOG_PATH=$(DEMO_LOG) $(PYTHON) -m uvicorn src.inference.api:app --host 127.0.0.1 --port 8000

ui-demo:
	CREDIT_RISK_MODEL_DIR=$(DEMO_MODELS) CREDIT_RISK_LOG_PATH=$(DEMO_LOG) $(PYTHON) -m streamlit run src/dashboard/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m ruff format --check src scripts tests

features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.models.train
