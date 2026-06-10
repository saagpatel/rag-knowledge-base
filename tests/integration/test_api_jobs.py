"""Integration tests — /api/jobs background ingest job tracking."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


def _unique_name() -> str:
    return f"jobs_{uuid.uuid4().hex[:8]}"


async def test_directory_ingest_returns_job(http_client, tmp_path):
    """POST /api/ingest with directory → 200, returns job_id + status=running."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    # Create test files
    for i in range(3):
        f = tmp_path / f"job_doc_{i}.md"
        f.write_text(f"# Job Document {i}\n\nContent for job test {i}.")

    resp = await http_client.post(
        "/api/ingest",
        json={"path": str(tmp_path), "collection": col},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "job_id" in data
    assert data["status"] == "running"
    assert data["total_files"] == 3

    # Wait for job to complete before cleanup
    job_id = data["job_id"]
    for _ in range(30):
        jr = await http_client.get(f"/api/jobs/{job_id}")
        if jr.json()["data"]["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.5)

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")


async def test_job_status_polling(http_client, tmp_path):
    """GET /api/jobs/{job_id} → returns status fields."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    f = tmp_path / "poll_doc.md"
    f.write_text("# Poll Test\n\nSimple document for polling test.")

    resp = await http_client.post(
        "/api/ingest",
        json={"path": str(tmp_path), "collection": col},
    )
    job_id = resp.json()["data"]["job_id"]

    # Poll job
    jr = await http_client.get(f"/api/jobs/{job_id}")
    assert jr.status_code == 200
    job_data = jr.json()["data"]
    assert job_data["job_id"] == job_id
    assert job_data["status"] in ("running", "completed", "failed")
    assert "total_files" in job_data
    assert "processed_files" in job_data

    # Wait for completion before cleanup
    for _ in range(30):
        jr = await http_client.get(f"/api/jobs/{job_id}")
        if jr.json()["data"]["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.5)

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")


async def test_job_completes_with_correct_count(http_client, tmp_path):
    """Poll job until completed → processed_files matches file count."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    for i in range(2):
        f = tmp_path / f"count_doc_{i}.md"
        f.write_text(f"# Count Doc {i}\n\nDocument number {i} for count test.")

    resp = await http_client.post(
        "/api/ingest",
        json={"path": str(tmp_path), "collection": col},
    )
    job_id = resp.json()["data"]["job_id"]

    # Poll until completed
    final_status = None
    for _ in range(60):
        jr = await http_client.get(f"/api/jobs/{job_id}")
        job_data = jr.json()["data"]
        if job_data["status"] in ("completed", "failed"):
            final_status = job_data
            break
        await asyncio.sleep(0.5)

    assert final_status is not None, "Job did not complete within timeout"
    assert final_status["status"] == "completed"
    assert final_status["processed_files"] == 2

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")


async def test_job_not_found(http_client):
    """GET /api/jobs/nonexistent → 404."""
    resp = await http_client.get("/api/jobs/nonexistent_job_id_999")
    assert resp.status_code == 404
