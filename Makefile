.PHONY: setup sample data features train evaluate holdout model-card drift shap hyperparam benchmark-ulb dashboard api test lint all clean

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

holdout:          ## Evaluate XGBoost on external hold-out (fraudTest.csv) -> reports/holdout_eval.json
	python -m sentinel.holdout

model-card:       ## Generate reports/model_card.md from evaluation.json
	python .kiro/skills/model-card/scripts/generate_model_card.py \
		reports/evaluation.json reports/model_card.md

quant-risk:       ## VaR/ES + out-of-time backtest -> reports/quant_risk.json
	python -m sentinel.risk_quant.quant_report

hyperparam:       ## XGBoost hyperparameter search (PR-AUC) -> reports/hyperparam_search.json
	python -m sentinel.hyperparam

benchmark-ulb:    ## Benchmark on ULB PCA creditcard dataset -> reports/benchmark_ulb.json
	python scripts/benchmark_ulb.py

drift:            ## Production drift monitor -> reports/drift.json
	python -m sentinel.drift

shap:             ## SHAP explainability -> reports/shap_summary.json + plots
	python .kiro/skills/shap-explainability/scripts/generate_shap.py

copilot:          ## Run the RAG + agent investigation demo -> reports/investigations.json
	python -m sentinel.copilot.demo

copilot-eval:     ## Evaluate copilot agreement with labels -> reports/copilot_evaluation.json
	python -m sentinel.copilot.evaluate_copilot

dashboard:        ## Launch the Streamlit executive dashboard
	streamlit run src/sentinel/serving/dashboard.py

api:              ## Launch the FastAPI scoring service
	uvicorn sentinel.serving.api:app --reload --app-dir src

test:             ## Run tests with coverage
	PYTHONPATH=src pytest --cov=src/sentinel --cov-report=term-missing

lint:             ## Ruff lint
	ruff check src tests

all: sample train evaluate quant-risk drift shap copilot model-card   ## End-to-end on the sample

clean:
	rm -rf models reports/*.json reports/*.md .pytest_cache .ruff_cache htmlcov .coverage
