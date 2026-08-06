# Shifa42 — common tasks.
#
# The pipeline targets are ordered by dependency: `data` must precede `index`,
# which must precede `graph` and `bench`.

.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := backend/.venv/bin/python

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ setup

.PHONY: install
install: install-backend install-frontend ## Install everything

.PHONY: install-backend
install-backend: ## Create the venv and install backend deps
	@command -v brew >/dev/null && brew list libomp >/dev/null 2>&1 \
		|| echo "NOTE: macOS needs 'brew install libomp' for xgboost."
	cd backend && uv venv --python 3.11 && uv pip install -e ".[dev]"

.PHONY: install-frontend
install-frontend: ## Install frontend deps
	cd frontend && npm install

# ------------------------------------------------------------------ pipeline

.PHONY: data
data: ## Ingest guidelines, MIMIC-IV-ED demo and RxNorm
	cd backend && ../$(PY) scripts/ingest_guidelines.py
	cd backend && ../$(PY) scripts/ingest_mimic.py
	cd backend && ../$(PY) scripts/ingest_rxnorm.py

.PHONY: index
index: ## Embed chunks and build the hybrid index (resumable; 45-90 min)
	cd backend && ../$(PY) scripts/embed_index.py

.PHONY: risk
risk: ## Train and evaluate the ED risk model
	cd backend && ../$(PY) scripts/train_risk.py

.PHONY: graph
graph: ## Build the knowledge graph (requires GROQ_API_KEY)
	cd backend && ../$(PY) scripts/build_graph.py

.PHONY: diag
diag: ## Retrieval diagnostics — no API key needed. Override: make diag N=200
	cd backend && ../$(PY) scripts/eval_retrieval.py --n $(or $(N),100)

.PHONY: bench
bench: ## Run benchmarks (requires GROQ_API_KEY). Override: make bench N=200
	cd backend && ../$(PY) scripts/run_benchmarks.py --n $(or $(N),50)

.PHONY: pipeline
pipeline: data index risk diag ## Everything that needs no API key

# ------------------------------------------------------------------ run

.PHONY: api
api: ## Run the FastAPI backend
	cd backend && ../$(PY) -m uvicorn app.main:app --reload --port 8000

.PHONY: web
web: ## Run the Next.js frontend
	cd frontend && npm run dev

# ------------------------------------------------------------------ quality

.PHONY: check
check: lint types test build ## Everything CI runs

.PHONY: lint
lint: ## Lint both packages
	cd backend && ../$(PY) -m ruff check app scripts tests
	cd frontend && npm run lint

.PHONY: types
types: ## Type-check both packages
	cd backend && ../$(PY) -m mypy app || true
	cd frontend && npx tsc --noEmit

.PHONY: test
test: ## Run the backend test suite
	cd backend && ../$(PY) -m pytest tests -q

.PHONY: build
build: ## Production build of the frontend
	cd frontend && npm run build

.PHONY: fmt
fmt: ## Auto-fix lint issues
	cd backend && ../$(PY) -m ruff check app scripts tests --fix

# ------------------------------------------------------------------ housekeeping

.PHONY: clean
clean: ## Remove caches and build output (keeps ingested data)
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/.ruff_cache backend/.mypy_cache
	rm -rf frontend/.next
