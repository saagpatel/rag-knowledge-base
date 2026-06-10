.PHONY: setup dev dev-strict dev-mcp dev-web build-web test test-integration test-cov lint format type-check docker-up docker-down docker-logs docker-test-up docker-test-down prod-build prod-up prod-down prod-logs init-db clean check health logs-api reset-db

setup:
	uv sync

dev:
	uv run uvicorn rag_kb.api:app --host 127.0.0.1 --port 8000 --reload

dev-strict:
	RAG_SERVER__STRICT_STARTUP=true uv run uvicorn rag_kb.api:app --host 127.0.0.1 --port 8000 --reload

dev-mcp:
	uv run rag-kb-mcp

dev-web:
	cd web && npm run dev

build-web:
	cd web && npm run build

test:
	uv run pytest

test-integration:
	uv run pytest tests/integration/ -v -m integration

test-cov:
	uv run pytest --cov=src/rag_kb

lint:
	uv run ruff check src/ tests/

format:
	uv run black src/ tests/ && uv run isort src/ tests/

type-check:
	uv run mypy src/

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-test-up:
	docker compose -f docker-compose.yml -f docker-compose.test.yml up -d

docker-test-down:
	docker compose -f docker-compose.yml -f docker-compose.test.yml down -v

prod-build:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build

prod-up:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

prod-down:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml down

prod-logs:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

init-db:
	uv run python -c "import asyncio; from rag_kb.core.database import init_db; asyncio.run(init_db())"

check: lint type-check test

health:
	curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool

logs-api:
	tail -f logs/rag_kb.log | python3 -m json.tool

reset-db:
	rm -f data/rag_kb.db && $(MAKE) init-db

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist *.egg-info
