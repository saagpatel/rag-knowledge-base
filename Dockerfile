# ============================================================================
# Stage 1: Build React web UI
# ============================================================================
FROM node:22-slim AS web-builder

WORKDIR /build
COPY web/package.json web/package-lock.json* ./
RUN npm ci --ignore-scripts
COPY web/ ./
RUN npm run build

# ============================================================================
# Stage 2: Python runtime with uv
# ============================================================================
FROM python:3.12-slim AS runtime

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

# Copy application source
COPY src/ src/
COPY config.yaml ./
COPY migrations/ migrations/

# Copy built web assets
COPY --from=web-builder /build/dist/ web/dist/

# Create data and log directories
RUN mkdir -p data logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

# Run with uvicorn — bind to 0.0.0.0 inside container (Nginx fronts it)
CMD ["uv", "run", "uvicorn", "rag_kb.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
