# ==============================================================================
# Production Dockerfile for Real-Time Object Detection & Tracking Backend
# Multi-stage / optimized slim build with non-root security and healthcheck
# ==============================================================================

FROM python:3.11-slim-bookworm AS runtime

# Set environment variables for Python execution and pip performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEVICE=cpu

# Install essential system runtime libraries for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application workspace
WORKDIR /app

# Create non-root user and group (UID/GID 1000)
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

# Pre-create operational data directories with correct permissions
RUN mkdir -p /app/data/uploads /app/outputs /app/logs /app/models/weights \
    /app/models/custom /app/models/pretrained /app/models/registry /app/models/metadata \
    && chown -R appuser:appuser /app

# Copy dependency definition and install packages as root
COPY docker/requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend application source code, configuration, and registry
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser yolov8n.pt /app/yolov8n.pt

# Switch to non-root execution user for security hardening
USER appuser

# Expose backend REST & WebSocket port
EXPOSE 8000

# Container healthcheck using root /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch ASGI server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
