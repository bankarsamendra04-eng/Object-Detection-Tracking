# Alias for docker/backend.Dockerfile to support root-level docker build
# Usage: docker build -t vision-backend .
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

RUN mkdir -p /app/data/uploads /app/outputs /app/logs /app/models/weights \
    /app/models/custom /app/models/pretrained /app/models/registry /app/models/metadata \
    && chown -R appuser:appuser /app

COPY docker/requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser yolov8n.pt /app/yolov8n.pt

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
