from fastapi.testclient import TestClient
import joblib
import os
from app import app

client = TestClient(app)

def test_model_file_exists():
    assert os.path.exists('crop_yield_model.pkl'), "Model file missing!"

def test_api_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is online"}

def test_api_predict_endpoint():
    payload = {
        "rainfall_mm": 650.0,
        "avg_temp_c": 24.5,
        "soil_ph": 6.5,
        "nitrogen_ppm": 80.0,
        "ndvi_index": 0.65
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_yield_kg_ha" in data
    assert isinstance(data["predicted_yield_kg_ha"], float)
