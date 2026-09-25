import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


class TestModelsAPI:
    """Integration tests for the /api/v1/models endpoints."""

    def test_list_models_endpoint(self):
        """GET /api/v1/models returns registered models and status."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "active_model_id" in data
        assert "total_models" in data
        assert data["total_models"] >= 2
        assert isinstance(data["models"], list)

        # Check general_pretrained is present and available
        pretrained = next((m for m in data["models"] if m["model_id"] == "general_pretrained"), None)
        assert pretrained is not None
        assert pretrained["status"] in {"AVAILABLE", "LOADED"}

        # Check custom_visdrone is present and NOT_AVAILABLE
        visdrone = next((m for m in data["models"] if m["model_id"] == "custom_visdrone"), None)
        assert visdrone is not None
        assert visdrone["status"] == "NOT_AVAILABLE"
        assert visdrone["input_size"] == 1280

    def test_get_active_model_endpoint(self):
        """GET /api/v1/models/active returns active model metadata."""
        response = client.get("/api/v1/models/active")
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert "model_name" in data
        assert data["is_active"] is True

    def test_get_model_by_id_endpoint(self):
        """GET /api/v1/models/{model_id} returns specific model information."""
        response = client.get("/api/v1/models/general_pretrained")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "general_pretrained"
        assert data["model_type"] == "pretrained"

        # Non-existent model returns 404
        not_found = client.get("/api/v1/models/unknown_model_xyz")
        assert not_found.status_code == 404

    def test_validate_pretrained_model_endpoint(self):
        """POST /api/v1/models/general_pretrained/validate validates available weights."""
        response = client.post("/api/v1/models/general_pretrained/validate")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "general_pretrained"
        assert data["status"] == "VALID"
        assert data["available"] is True
        assert data["loadable"] is True
        assert data["inference_tested"] is True

    def test_validate_unavailable_model_endpoint(self):
        """POST /api/v1/models/custom_visdrone/validate reports NOT_AVAILABLE."""
        response = client.post("/api/v1/models/custom_visdrone/validate")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "custom_visdrone"
        assert data["status"] == "NOT_AVAILABLE"
        assert data["available"] is False
        assert data["loadable"] is False

    def test_switch_model_endpoint_success(self):
        """POST /api/v1/models/general_pretrained/switch successfully activates model."""
        response = client.post("/api/v1/models/general_pretrained/switch")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["active_model_id"] == "general_pretrained"

    def test_switch_unavailable_model_endpoint_failure(self):
        """POST /api/v1/models/custom_visdrone/switch fails with 400 because weights are absent."""
        response = client.post("/api/v1/models/custom_visdrone/switch")
        assert response.status_code == 400
        data = response.json()
        detail = data["detail"].lower()
        assert "not available" in detail or "does not exist" in detail

    def test_switch_unregistered_model_endpoint_failure(self):
        """POST /api/v1/models/fake_model/switch fails with 404."""
        response = client.post("/api/v1/models/fake_model/switch")
        assert response.status_code == 404
