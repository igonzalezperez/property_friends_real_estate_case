import json
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

CLIENT = TestClient(app)

ENDPOINT = "/api/v1/health"
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


def test_health_success():
    with patch("app.api.routes.predictor.get_prediction"):
        response = CLIENT.get(ENDPOINT, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"status": True}


def test_health_unauthorized():
    response = CLIENT.get(ENDPOINT)  # No headers provided
    assert response.status_code == 403
    assert response.json() == {"detail": "Not authenticated"}


def test_health_unavailable():
    os.environ["ML_MODELS_DIR"] = "fake_dir"  # Point to nonexistent model
    response = CLIENT.get(ENDPOINT, headers=HEADERS)
    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable"}
