.DEFAULT_GOAL := help
.PHONY: help setup db-up db-down db-reset migrate ingest manifest \
        lint format typecheck test test-integration test-network check run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and pre-commit hooks
	uv sync --all-extras
	uv run pre-commit install

db-up: ## Start Postgres + pgvector (docker compose)
	docker compose up -d postgres
	@echo "waiting for postgres..."
	@until docker exec trialrag-postgres pg_isready -U trialrag -d trialrag >/dev/null 2>&1; do sleep 1; done
	@echo "postgres is ready"

db-down: ## Stop the local stack
	docker compose down

db-reset: db-up ## Drop and recreate the local schema, then re-migrate
	docker exec trialrag-postgres psql -U trialrag -d trialrag \
		-c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) migrate

migrate: ## Apply pending SQL migrations
	uv run trialrag migrate-db

ingest: ## Run the full pipeline (fetch -> parse -> chunk -> embed -> load)
	uv run trialrag ingest $(if $(LIMIT),--limit $(LIMIT),)

manifest: ## Regenerate docs/corpus_manifest.json from current DB state
	uv run trialrag manifest

lint: ## Ruff check
	uv run ruff check .

format: ## Ruff format (writes changes)
	uv run ruff format .

typecheck: ## mypy --strict
	uv run mypy

test: ## Unit tests only (hermetic, no network, no DB required)
	uv run pytest tests/ -m "not network and not integration"

test-integration: db-up ## Unit + integration tests (needs local Postgres)
	uv run pytest tests/ -m "not network"

test-network: ## Full suite including the live ClinicalTrials.gov contract test
	uv run pytest tests/

check: lint typecheck test-integration ## Everything CI runs on a PR

run: ## Start the API locally (http://localhost:8000)
	uv run uvicorn trialrag.api.app:app --reload --port 8000

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis htmlcov .coverage
