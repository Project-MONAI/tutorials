"""
API Endpoint Tests

Tests for FastAPI endpoints including health checks and prediction.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self):
        """Test that root endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_api_info(self):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["name"] == "MONAI Inference API"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self):
        """Test that health endpoint returns status information."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    def test_health_status_format(self):
        """Test that health response has expected format."""
        response = client.get("/health")
        data = response.json()

        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["device"], str)
        assert data["status"] in ["healthy", "model_not_loaded"]


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_requires_file(self):
        """Test that predict endpoint requires a file."""
        response = client.post("/predict")
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_rejects_invalid_format(self):
        """Test that predict endpoint rejects non-NIfTI files."""
        # Create a fake file with wrong extension
        files = {"file": ("test.txt", b"fake content", "text/plain")}
        response = client.post("/predict", files=files)

        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]

    def test_predict_accepts_nifti_extension(self):
        """Test that predict endpoint accepts .nii files."""
        # Note: This will fail inference if model not loaded,
        # but should pass file validation
        files = {"file": ("test.nii", b"fake nifti data", "application/octet-stream")}
        response = client.post("/predict", files=files)

        # Should get past file validation (not 400)
        # May get 503 (model not loaded) or 500 (invalid data)
        assert response.status_code in [500, 503]


class TestDocumentation:
    """Tests for API documentation endpoints."""

    def test_docs_available(self):
        """Test that Swagger UI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """Test that ReDoc documentation is available."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses."""
        response = client.get("/health")

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
