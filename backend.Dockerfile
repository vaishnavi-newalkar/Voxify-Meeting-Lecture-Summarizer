# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production image ─────────────────────────────────────────────────
FROM python:3.11-slim

# Security: non-root user
RUN groupadd -r voxify && useradd -r -g voxify -d /app -s /sbin/nologin voxify

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY api.py .
COPY utils/ ./utils/

# Own everything by non-root user
RUN chown -R voxify:voxify /app

USER voxify

# Render uses PORT env var (defaults to 10000); local Docker uses 8000
ENV PORT=8000

EXPOSE ${PORT}

# Health check — uses the PORT env var
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health')" || exit 1

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 2
