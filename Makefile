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

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	uv run pytest

## Install git hooks (pre-commit + nbstripout notebook-output filter)
.PHONY: hooks
hooks:
	uv run pre-commit install --install-hooks
	uv run pre-commit install --hook-type pre-push
	uv run nbstripout --install --attributes .gitattributes

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

## Split folds and build features
.PHONY: features
features: data
	uv run jupyter execute --inplace 03_split_folds.ipynb
	uv run jupyter execute --inplace 04_feature_engineering.ipynb

## Train the persistence baseline
.PHONY: train-persistence
train-persistence: features
	uv run jupyter execute --inplace 05_train_persistence.ipynb

## Train the Ridge regression model
.PHONY: train-ridge
train-ridge: features
	uv run jupyter execute --inplace 05_train_ridge.ipynb

## Train all models
.PHONY: train
train: train-persistence train-ridge

## Evaluate trained models (assumes `make train` has already been run)
.PHONY: evaluate
evaluate:
	uv run jupyter execute --inplace 06_evaluate.ipynb

## Select the winning model/run
.PHONY: model_selection
model_selection: evaluate
	uv run jupyter execute --inplace 07_model_selection.ipynb

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
