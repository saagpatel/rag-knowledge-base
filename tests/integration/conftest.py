"""Integration test fixtures — skip guards + real-service session fixtures."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from rag_kb.api import create_app
from rag_kb.core.config import AppConfig, OllamaConfig, QdrantConfig, SqliteConfig
from rag_kb.core.database import init_db
from rag_kb.core.ollama_client import OllamaClient
from rag_kb.core.qdrant_client import QdrantManager

# ---------------------------------------------------------------------------
# Skip guards — evaluated once at import time
# ---------------------------------------------------------------------------


def _qdrant_up() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:6333/healthz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_up() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


services_available = pytest.mark.skipif(
    not (_qdrant_up() and _ollama_up()),
    reason="Qdrant or Ollama not running",
)

# ---------------------------------------------------------------------------
# Session-scoped fixtures — shared across all integration tests
# ---------------------------------------------------------------------------

COLLECTION_PREFIX = "test_integ_"
MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"


@pytest.fixture(scope="session")
def integration_config(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    """Config pointing at real services but with isolated collection prefix + temp DB."""
    tmp = tmp_path_factory.mktemp("integ")
    return AppConfig(
        ollama=OllamaConfig(),
        qdrant=QdrantConfig(collection_prefix=COLLECTION_PREFIX),
        sqlite=SqliteConfig(path=str(tmp / "integ_test.db")),
    )


@pytest_asyncio.fixture(scope="session")
async def integ_db(integration_config: AppConfig):
    """Real SQLite connection with migrations applied."""
    db = await init_db(
        db_path=integration_config.sqlite.path,
        migrations_dir=str(MIGRATIONS_DIR),
    )
    yield db
    await db.close()


@pytest_asyncio.fixture(scope="session")
async def integ_ollama():
    """Real OllamaClient, session-scoped to share embedding connections."""
    async with OllamaClient() as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def integ_qdrant(integration_config: AppConfig):
    """Real QdrantManager. Teardown deletes all test_integ_* collections."""
    mgr = QdrantManager(config=integration_config.qdrant)
    yield mgr
    # Cleanup: delete all test collections
    try:
        collections = await mgr.list_collections()
        for name in collections:
            if COLLECTION_PREFIX in name:
                try:
                    await mgr._client.delete_collection(name)
                except Exception:
                    pass
    except Exception:
        pass
    await mgr.close()


@pytest_asyncio.fixture(scope="session")
async def api_app(integ_db, integ_ollama, integ_qdrant):
    """FastAPI app with real services injected (bypasses lifespan)."""
    application = create_app()
    # Inject real services into app.state, bypassing the lifespan
    application.state.db = integ_db
    application.state.ollama = integ_ollama
    application.state.qdrant = integ_qdrant
    application.state.start_time = time.monotonic()
    return application


@pytest_asyncio.fixture(scope="session")
async def http_client(api_app):
    """httpx AsyncClient with ASGITransport — full HTTP stack, no real server."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Per-test helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def test_collection(integ_qdrant: QdrantManager):
    """Create a unique collection, yield its name, delete on teardown."""
    name = f"col_{uuid.uuid4().hex[:8]}"
    await integ_qdrant.create_collection(name)
    yield name
    try:
        await integ_qdrant.delete_collection(name)
    except Exception:
        pass


@pytest.fixture
def sample_doc(tmp_path: Path) -> Path:
    """Write a sample Markdown file and return its path."""
    doc = tmp_path / "sample_doc.md"
    doc.write_text(
        "# Vector Databases\n\n"
        "Vector databases store high-dimensional embeddings for similarity search.\n"
        "They are essential for RAG pipelines and semantic retrieval.\n\n"
        "## Key Features\n\n"
        "- Fast approximate nearest neighbor search\n"
        "- Support for metadata filtering\n"
        "- Horizontal scalability\n"
    )
    return doc
