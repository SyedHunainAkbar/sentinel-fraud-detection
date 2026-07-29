.PHONY: setup sample data features train evaluate model-card dashboard api test lint all clean

setup:            ## Install dependencies
	pip install -r requirements.txt

sample:           ## Regenerate the tiny synthetic sample (for tests/CI)
	python scripts/generate_sample.py

data:             ## Download the full Sparkov dataset from Kaggle
	bash scripts/download_data.sh

train:            ## Train models and persist artifacts
	python -m sentinel.train

evaluate:         ## Compute metrics + cost-optimal threshold -> reports/evaluation.json
	python -m sentinel.evaluate

model-card:       ## Generate reports/model_card.md from evaluation.json
	python .kiro/skills/model-card/scripts/generate_model_card.py \
		reports/evaluation.json reports/model_card.md

quant-risk:       ## VaR/ES + out-of-time backtest -> reports/quant_risk.json
	python -m sentinel.risk_quant.quant_report

copilot:          ## Run the RAG + agent investigation demo -> reports/investigations.json
	python -m sentinel.copilot.demo

dashboard:        ## Launch the Streamlit executive dashboard
	streamlit run src/sentinel/serving/dashboard.py

api:              ## Launch the FastAPI scoring service
	uvicorn sentinel.serving.api:app --reload --app-dir src

test:             ## Run tests with coverage
	PYTHONPATH=src pytest --cov=src/sentinel --cov-report=term-missing

lint:             ## Ruff lint
	ruff check src tests

all: sample train evaluate quant-risk copilot model-card   ## End-to-end on the sample

clean:
	rm -rf models reports/*.json reports/*.md .pytest_cache .ruff_cache htmlcov .coverage
