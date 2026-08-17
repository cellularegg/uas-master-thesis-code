#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = uas-master-thesis-code-notebooks
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all local MLflow tracking data (requires two confirmations)
.PHONY: clean-mlflow
clean-mlflow:
	@printf '%s\n' 'WARNING: this permanently deletes mlflow.db and mlruns/.'; \
	printf '%s' "First confirmation: type 'yes' to continue: "; \
	read -r confirmation; \
	if [ "$$confirmation" != "yes" ]; then \
		printf '%s\n' 'Aborted.'; \
		exit 1; \
	fi; \
	printf '%s' "Second confirmation: type 'yes' to permanently delete the data: "; \
	read -r confirmation; \
	if [ "$$confirmation" != "yes" ]; then \
		printf '%s\n' 'Aborted.'; \
		exit 1; \
	fi; \
	rm -rf -- mlruns; \
	rm -f -- mlflow.db; \
	printf '%s\n' 'MLflow data deleted.'

## Lint using ruff and mypy (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check
	uv run mypy src tests

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

## Run tests
.PHONY: test
test:
	uv run pytest

## Install git hooks (pre-commit + pre-push)
.PHONY: hooks
hooks:
	uv run pre-commit install --install-hooks
	uv run pre-commit install --hook-type pre-push

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Fetch and preprocess data
.PHONY: data
data: requirements
	uv run jupyter execute --inplace 01_fetch_data.ipynb
	uv run jupyter execute --inplace 02_preprocessing.ipynb

## Build features (the data dependency re-runs fetch and preprocessing first)
.PHONY: features
features: data
	uv run jupyter execute --inplace 03_feature_engineering.ipynb

## Train the persistence baseline
.PHONY: train-persistence
train-persistence: features
	uv run jupyter execute --inplace 04_01_train_persistence.ipynb

## Train the Ridge regression model
.PHONY: train-ridge
train-ridge: features
	uv run jupyter execute --inplace 04_02_train_ridge.ipynb

## Train the MLP model
.PHONY: train-mlp
train-mlp: features
	uv run jupyter execute --inplace 04_03_train_mlp.ipynb

## Train all models
.PHONY: train
train: train-persistence train-ridge train-mlp

## Evaluate trained models (assumes `make train` has already been run)
.PHONY: evaluate
evaluate:
	uv run jupyter execute --inplace 05_evaluate.ipynb

## Select the winning model/run
.PHONY: model_selection
model_selection: evaluate
	uv run jupyter execute --inplace 06_model_selection.ipynb

## Launch the MLflow UI backed by a local SQLite store
.PHONY: mlflowui
mlflowui:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
