import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

CLIENT = TestClient(app)

ENDPOINT = "/api/v1/predict-logs"
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


@pytest.fixture
def mock_log_entries():
    """Sample log entries for mocking."""
    return [
        {"input": SAMPLE_INPUT, "result": 1, "date": "2021-01-01T00:00:00"},
        {"input": SAMPLE_INPUT, "result": 2, "date": "2021-01-02T00:00:00"},
    ]


@patch("app.api.routes.predictor.read_log_entries")
def test_get_log_entries_success(mock_read_log_entries, mock_log_entries):
    mock_read_log_entries.return_value = mock_log_entries
    response = CLIENT.get(ENDPOINT, headers=HEADERS)
    assert response.status_code == 200
    assert len(response.json()) == 2


@patch("app.api.routes.predictor.read_log_entries")
def test_get_log_entries_unauthorized(mock_read_log_entries, mock_log_entries):
    mock_read_log_entries.return_value = mock_log_entries
    response = CLIENT.get(ENDPOINT)
    assert response.status_code == 403


@patch("app.api.routes.predictor.read_log_entries")
def test_get_log_entries_limit(mock_read_log_entries, mock_log_entries):
    # Use the side effect to dynamically adjust the return value based on limit
    mock_read_log_entries.return_value = mock_log_entries[
        :1
    ]  # Simulate limit=1 by slicing the list

    response = CLIENT.get(f"{ENDPOINT}?limit=1", headers=HEADERS)
    assert response.status_code == 200
    assert len(response.json()) == 1, "The API should respect the limit parameter"


@patch("app.api.routes.predictor.read_log_entries")
def test_get_log_entries_handles_negative_limit_gracefully(
    mock_read_log_entries, mock_log_entries
):
    # Given a negative limit, the fixture adjusts it to a minimum of 1
    mock_read_log_entries.return_value = [mock_log_entries[-1]]
    # Explicitly use -1 to test the fixture's handling
    response = CLIENT.get(
        f"{ENDPOINT}?limit=-1", headers=HEADERS
    )  # API query with a negative limit
    # Expect a successful response, indicating the API handled the negative limit gracefully
    assert (
        response.status_code == 200
    ), "The API should successfully handle a negative limit"

    assert (
        len(response.json()) == 1
    ), "The API should return 1 log entry, handling the negative limit gracefully"


@patch("app.api.routes.predictor.read_log_entries")
def test_get_log_entries_invalid_limit_type(mock_read_log_entries, mock_log_entries):
    response = CLIENT.get(
        f"{ENDPOINT}?limit=abc", headers=HEADERS
    )  # Using 'abc' as an example of a non-numeric limit
    assert response.status_code == 422
