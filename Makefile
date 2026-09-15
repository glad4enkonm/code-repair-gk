.PHONY: help install install-dev test test-slow test-all test-cov lint format format-check type-check clean all-checks

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in editable mode"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run tests (skips tests marked slow)"
	@echo "  test-slow    - Run only the tests marked slow"
	@echo "  test-all     - Run all tests including slow ones"
	@echo "  test-cov     - Run tests (skips slow) with coverage"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black)"
	@echo "  format-check - Check formatting without modifying"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  all-checks   - Run all quality checks (fast: skips slow tests)"
	@echo "  clean        - Clean build artifacts"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing targets
test:
	pytest -m "not slow"

test-slow:
	pytest -m "slow"

test-all:
	pytest

test-cov:
	pytest -m "not slow" --cov=kg --cov-report=term-missing

# Code quality targets
lint:
	flake8 src/ tests/

format:
	black src/ tests/

format-check:
	black --check src/ tests/

type-check:
	mypy -p kg

# Combined quality check
all-checks: format-check lint type-check test

# Build and distribution targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
