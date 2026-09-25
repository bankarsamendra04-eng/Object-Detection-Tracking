"""
Step 16 Automated Docker and Container Deployment Validation Test Suite.
Validates Dockerfiles, Docker Compose configs, Nginx routing, and .dockerignore rules.
"""

from pathlib import Path
import re
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDockerConfiguration:
    """Test suite validating Docker and Compose configurations."""

    def test_docker_files_exist(self):
        """Verify all required Docker artifacts exist on disk."""
        required_files = [
            PROJECT_ROOT / "docker" / "backend.Dockerfile",
            PROJECT_ROOT / "docker" / "frontend.Dockerfile",
            PROJECT_ROOT / "docker" / "nginx.conf",
            PROJECT_ROOT / "docker" / "requirements-docker.txt",
            PROJECT_ROOT / "docker" / "requirements-gpu.txt",
            PROJECT_ROOT / "docker-compose.yml",
            PROJECT_ROOT / "docker-compose.gpu.yml",
            PROJECT_ROOT / "Dockerfile",
            PROJECT_ROOT / "Dockerfile.frontend",
            PROJECT_ROOT / ".dockerignore",
            PROJECT_ROOT / "frontend" / ".dockerignore",
            PROJECT_ROOT / ".env.example",
            PROJECT_ROOT / ".env.docker.example",
            PROJECT_ROOT / "docs" / "DOCKER.md",
        ]
        for f in required_files:
            assert f.is_file(), f"Required Docker file missing: {f}"

    def test_docker_compose_yaml_valid(self):
        """Validate docker-compose.yml structure, services, and healthchecks."""
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "services" in data, "docker-compose.yml must contain 'services'"
        services = data["services"]

        # Validate backend service
        assert "backend" in services, "Must define 'backend' service"
        backend = services["backend"]
        assert "build" in backend
        assert "ports" in backend
        assert "environment" in backend
        assert "volumes" in backend
        assert "healthcheck" in backend
        assert "networks" in backend

        # Check backend healthcheck
        hc = backend["healthcheck"]
        assert "test" in hc
        assert "/health" in str(hc["test"])

        # Validate frontend service
        assert "frontend" in services, "Must define 'frontend' service"
        frontend = services["frontend"]
        assert "build" in frontend
        assert "ports" in frontend
        assert "depends_on" in frontend
        assert "backend" in frontend["depends_on"]
        assert "healthcheck" in frontend
        assert "networks" in frontend

        # Validate named volumes
        assert "volumes" in data
        assert "vision-backend-data" in data["volumes"]
        assert "vision-backend-outputs" in data["volumes"]
        assert "vision-backend-logs" in data["volumes"]

        # Validate networks
        assert "networks" in data
        assert "vision-network" in data["networks"]

    def test_docker_compose_gpu_yaml_valid(self):
        """Validate docker-compose.gpu.yml structure."""
        gpu_compose_path = PROJECT_ROOT / "docker-compose.gpu.yml"
        with open(gpu_compose_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "services" in data
        assert "backend" in data["services"]
        backend = data["services"]["backend"]
        assert "deploy" in backend
        assert "resources" in backend["deploy"]

    def test_backend_dockerfile_best_practices(self):
        """Verify backend Dockerfile implements container security and best practices."""
        dockerfile_path = PROJECT_ROOT / "docker" / "backend.Dockerfile"
        content = dockerfile_path.read_text(encoding="utf-8")

        # Must use slim Python 3.11 base
        assert "FROM python:3.11-slim" in content, "Must use python:3.11-slim base image"

        # Must set recommended Python environment variables
        assert "PYTHONDONTWRITEBYTECODE=1" in content
        assert "PYTHONUNBUFFERED=1" in content

        # Must install required system packages for OpenCV/PyTorch
        assert "libgl1" in content
        assert "libglib2.0-0" in content
        assert "curl" in content

        # Must define non-root user
        assert "useradd" in content
        assert "USER appuser" in content

        # Must expose port 8000 and declare HEALTHCHECK
        assert "EXPOSE 8000" in content
        assert "HEALTHCHECK" in content
        assert "/health" in content

        # Must have ASGI entrypoint
        assert "uvicorn" in content
        assert "backend.app.main:app" in content

    def test_frontend_dockerfile_multi_stage(self):
        """Verify frontend Dockerfile utilizes multi-stage build."""
        dockerfile_path = PROJECT_ROOT / "docker" / "frontend.Dockerfile"
        content = dockerfile_path.read_text(encoding="utf-8")

        # Multi-stage validation: Stage 1 Node build, Stage 2 Nginx runtime
        assert "FROM node:20-alpine AS build" in content or "FROM node:" in content
        assert "FROM nginx:" in content
        assert "AS runtime" in content or "nginx" in content

        # Must install with lockfile and build
        assert "npm ci" in content
        assert "npm run build" in content

        # Must copy from build stage into nginx web root
        assert "--from=build" in content
        assert "/usr/share/nginx/html" in content

        # Must expose port 80 and have healthcheck
        assert "EXPOSE 80" in content
        assert "HEALTHCHECK" in content

    def test_nginx_configuration_routes(self):
        """Verify Nginx reverse proxy configuration handles all application paths."""
        nginx_path = PROJECT_ROOT / "docker" / "nginx.conf"
        content = nginx_path.read_text(encoding="utf-8")

        # Upstream definition
        assert "upstream backend_service" in content
        assert "backend:8000" in content

        # SPA Fallback
        assert "try_files $uri $uri/ /index.html" in content

        # WebSocket proxy with Upgrade headers
        assert "location /ws/" in content
        assert "proxy_pass http://backend_service" in content
        assert "proxy_set_header Upgrade $http_upgrade" in content
        assert 'proxy_set_header Connection "upgrade"' in content
        assert "proxy_read_timeout 3600s" in content
        assert "proxy_buffering off" in content

        # REST API proxy
        assert "location /api/" in content
        assert "client_max_body_size 60M" in content

        # Dedicated health endpoint
        assert "location = /healthz" in content or "location /healthz" in content

        # Security headers
        assert "X-Content-Type-Options" in content
        assert "X-Frame-Options" in content

    def test_dockerignore_rules(self):
        """Verify .dockerignore excludes virtualenvs, caches, and massive datasets."""
        dockerignore_path = PROJECT_ROOT / ".dockerignore"
        content = dockerignore_path.read_text(encoding="utf-8")

        assert ".venv" in content
        assert "node_modules" in content
        assert "__pycache__" in content
        assert ".git" in content
        assert "data/datasets/raw" in content
        assert "data/datasets/processed" in content
        assert ".env" in content

    def test_model_registry_compatibility(self):
        """Verify model weights and registry are properly aligned for container paths."""
        registry_path = PROJECT_ROOT / "models" / "registry" / "models.yaml"
        assert registry_path.is_file()
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)

        assert "models" in registry
        assert "general_pretrained" in registry["models"]
        general = registry["models"]["general_pretrained"]
        assert general["status"] == "AVAILABLE"
        # Confirm weights file exists in project
        weights_name = general["weights_path"]
        assert (PROJECT_ROOT / weights_name).is_file() or (PROJECT_ROOT / "models" / "weights" / weights_name).is_file()
