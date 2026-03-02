"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_churn_prediction_schema():
    """Test churn prediction endpoint with valid schema."""
    # Sample request
    request_data = {
        "customer_features": {
            "CustomerID": 12345,
            "recency_days": 30,
            "frequency": 10,
            "monetary_total": 1000,
            "monetary_mean": 100,
            "monetary_std": 20,
            "total_items": 50,
            "avg_items_per_order": 5,
            "max_items_per_order": 10,
            "avg_unit_price": 20,
            "max_unit_price": 50,
            "min_unit_price": 5,
            "price_range": 45,
            "tenure_days": 180,
            "active_days": 150,
            "purchase_velocity": 2.0,
            "avg_days_between_purchases": 15,
        }
    }
    
    # This will return 503 if model not trained, but validates schema
    response = client.post("/api/v1/churn/predict", json=request_data)
    
    # Either success or model not found (expected if not trained yet)
    assert response.status_code in [200, 503]


def test_batch_prediction_schema():
    """Test batch prediction endpoint schema."""
    request_data = {
        "customers": [
            {
                "CustomerID": 12345,
                "recency_days": 30,
                "frequency": 10,
                "monetary_total": 1000,
                "monetary_mean": 100,
                "monetary_std": 20,
                "total_items": 50,
                "avg_items_per_order": 5,
                "max_items_per_order": 10,
                "avg_unit_price": 20,
                "max_unit_price": 50,
                "min_unit_price": 5,
                "price_range": 45,
                "tenure_days": 180,
                "active_days": 150,
                "purchase_velocity": 2.0,
                "avg_days_between_purchases": 15,
            }
        ]
    }
    
    response = client.post("/api/v1/churn/predict-batch", json=request_data)
    assert response.status_code in [200, 503]


def test_invalid_request():
    """Test invalid request data."""
    invalid_data = {
        "customer_features": {
            "CustomerID": "invalid",  # Should be int
            "recency_days": "not_a_number",
        }
    }
    
    response = client.post("/api/v1/churn/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error
