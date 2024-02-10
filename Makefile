SHELL := /bin/bash

# Variables definitions
# -----------------------------------------------------------------------------

ifeq ($(TIMEOUT),)
TIMEOUT := 60
endif

ifeq ($(MODEL_PATH),)
MODEL_PATH := ./ml/model/
endif

ifeq ($(BASE_MODEL_NAME),)
BASE_MODEL_NAME := model.joblib
endif

# Target section and Global definitions
# -----------------------------------------------------------------------------
.PHONY: all clean test install run deploy down ci-pipeline ml-pipeline

all: clean test install run deploy down

ci-pipeline: ci-format ci-lint ci-security

ci-format:
	poetry run isort . --profile black --check-only
	poetry run black . --exclude .venv/ --check

ci-lint:
	poetry run flake8 --config=config/.flake8
	find . -name '*.py' ! -name '__init__.py' -exec poetry run pylint --rcfile=config/.pylintrc {} \;
	poetry run mypy --config-file=config/mypy.ini . --exclude=tests/

ci-security:
	poetry run bandit -ll -c config/.bandit.yml -r .

ml-pipeline:
	poetry run python ml/pipelines/luigi_tasks.py

ml-pipeline-base: make-dataset build-features train-model

make-dataset:
	poetry run python ml/pipelines/make_dataset.py

build-features:
	poetry run python ml/pipelines/build_features.py

train-model:
	poetry run python ml/pipelines/train_model.py

test:
	poetry run pytest tests -vv --show-capture=all

install: generate_dot_env
	pip install --upgrade pip
	pip install poetry
	poetry install --with dev --no-root

run:
	PYTHONPATH=app/ poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8080

deploy: generate_dot_env
	docker-compose build
	docker-compose up -d

down:
	docker-compose down

generate_dot_env:
	@if [[ ! -e .env ]]; then \
		cp .env.example .env; \
	fi

clean:
	@find . -name '*.pyc' -exec rm -rf {} \;
	@find . -name '__pycache__' -exec rm -rf {} \;
	@find . -name 'Thumbs.db' -exec rm -rf {} \;
	@find . -name '*~' -exec rm -rf {} \;
	rm -rf .cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf htmlcov
	rm -rf .tox/
	rm -rf docs/_build