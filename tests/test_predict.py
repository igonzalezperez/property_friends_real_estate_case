import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

CLIENT = TestClient(app)
ENDPOINT = "/api/v1/predict"
with open(Path("app", "valid_keys", "api_keys.json"), "r") as stream:
    api_keys = json.load(stream)
    API_KEY = api_keys["root"]["api_key"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
SAMPLE_INPUT = {
    "type": "departamento",
    "sector": "vitacura",
    "net_usable_area": 143.0,
    "net_area": 165.0,
    "n_rooms": 3.0,
    "n_bathroom": 4.0,
    "latitude": -33.40251,
    "longitude": -70.6610,
}


# Successful prediction
def test_predict_success():
    with patch("app.api.routes.predictor.get_prediction") as mock_get_prediction:
        mock_get_prediction.return_value = 0.5  # Mocked prediction result
        response = CLIENT.post(
            ENDPOINT,
            json=SAMPLE_INPUT,
            headers=HEADERS,
        )
    assert response.status_code == 200
    assert response.json() == {"prediction": 0.5}


# Invalid input data
def test_predict_invalid_input():
    response = CLIENT.post(ENDPOINT, json={}, headers=HEADERS)  # Empty input
    assert response.status_code == 422


# Authentication failure
def test_predict_unauthorized():
    response = CLIENT.post(ENDPOINT, json=SAMPLE_INPUT)  # No token provided
    assert response.status_code == 403


# Model not available
def test_predict_model_unavailable():
    import os

    os.environ["ML_MODELS_DIR"] = "fake_dir"
    response = CLIENT.post(ENDPOINT, json=SAMPLE_INPUT, headers=HEADERS)
    assert response.status_code == 503
    assert response.json() == {
        "detail": f"Exception: Machine learning model at {Path('fake_dir', 'trained_model', 'model.pkl')} doesn't exist",
    }
