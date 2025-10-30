.PHONY: help install test lint format clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run unit tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run code linters"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run evaluation in Docker"
	@echo "  make docker-dash   - Run dashboard in Docker"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=120
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/ --line-length=120

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

docker-build:
	docker-compose build

docker-run:
	docker-compose run --rm evaluator python -m src.pipeline --num-samples 100

docker-dash:
	docker-compose up dashboard

docker-down:
	docker-compose down

eval-quick:
	python -m src.pipeline --num-samples 10 --no-llm

eval-full:
	python run_full_evaluation.py

dashboard:
	streamlit run dashboard.py

regression-check:
	python scripts/check_regression.py \
		--baseline results/baseline.json \
		--current results/latest.json \
		--threshold 0.05

