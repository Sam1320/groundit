.DEFAULT_GOAL := all

.PHONY: .uv
.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit
.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: lint
lint: .pre-commit ## Lint the code
	uv run ruff format --check
	uv run ruff check

.PHONY: format
format: .uv ## Format the code
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: test
test: ## Run tests and collect coverage data
	uv run coverage run -m pytest
	@uv run coverage report

.PHONY: testcov
testcov: test ## Run tests and generate an HTML coverage report
	@echo "building coverage html"
	@uv run coverage html

.PHONY: typecheck
typecheck: ## Run type checks
	uv run ty check

.PHONY: all
all: format lint typecheck testcov ## Run all checks (default target)
