PACKAGE_DIR := src/inquiro

.DEFAULT_GOAL := help

.PHONY: help install build clean clean-build clean-cache

help: ## Show available targets
	@printf "\nInquiro development targets\n\n"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime dependencies
	uv pip install -e ".[dev]"

build: clean-build ## Build wheel and source distributions
	uv pip install --upgrade build
	uv python -m build

clean: clean-build clean-cache ## Remove generated files

clean-build: ## Remove build artifacts
	rm -rf build dist .eggs
	find . -maxdepth 2 -type d -name "*.egg-info" -exec rm -rf {} +

clean-cache: ## Remove Python cache files
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.egg*" \) -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
