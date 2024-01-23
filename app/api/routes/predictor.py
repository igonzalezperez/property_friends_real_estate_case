"""
This module defines the API routes and functions for making predictions and
checking the health of the model.

It includes the following routes:
- `/predict`: Takes input data, makes predictions, and returns the result.
- `/health`: Checks the health of the model by performing a test prediction.
- `/predict-logs`: Gets the last calls of /predict, including request body,
response and date of request.
"""
import datetime
import json
import typing
from pickle import load

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from numpy.typing import NDArray

from app.core.config import INPUT_EXAMPLE
from app.core.errors import PredictException
from app.core.middleware.api_key_middleware import token_auth_scheme
from app.core.monitoring import read_log_entries, save_to_json
from app.models.prediction import (
    HealthResponse,
    ModelInput,
    ModelResponse,
    PredictLogEntry,
)
from app.services.predict import ModelHandlerScore as model

router = APIRouter()
log_router = APIRouter()


@typing.no_type_check
def get_prediction(data_point: pd.DataFrame) -> NDArray[np.float64]:
    """
    Get predictions for the input data point. It performs a .predict() on
    a loaded model artifact, like an sklearn model or pipeline.

    :param pd.DataFrame data_point: The input data point for prediction.
    :return NDArray[np.float64]: Predicted value as an array of type
    np.float64.
    """
    return model.predict(
        data_point,
        load_wrapper=load,
        method="predict",
    )


@router.post(
    "/predict",
    response_model=ModelResponse,
    name="predict:get-inference",
    responses={
        403: {"description": "Forbidden"},
        503: {
            "description": "Model not available",
        },
    },
    dependencies=[Depends(token_auth_scheme)],  # api token authentication
)
async def predict(
    data_input: ModelInput, background_tasks: BackgroundTasks
) -> ModelResponse:
    """
    **Get predictions for the input data**

    This endpoint receives a ModelInput object and performs predictions on it.
    The payload and response fields are defined by ModelInput and ModelResponse

    **Input**
    - `data_input` (InputData): The input data for prediction.
    - `background_tasks` (BackgroundTasks): A background task for saving
    prediction results.

    **Output**
    - `ModelResponse`: The prediction results.

    **Raises**
    - `HTTPException 400`: If the 'data_input' argument is invalid.
    - `HTTPException 503`: If the model artifact doesn't exist or is
    unavailable.
    - `HTTPException 403`: If authentication fails (Forbidden).
    """
    if not data_input:
        raise HTTPException(
            status_code=400,
            detail="Bad request",
        )
    try:
        data_point = data_input.get_df()
        prediction = get_prediction(data_point)
        background_tasks.add_task(
            save_to_json,
            data_input=data_point,
            result=prediction,
        )
    except PredictException as err:
        raise HTTPException(status_code=503, detail=f"Exception: {err}") from err
    except FileNotFoundError as err:
        raise HTTPException(status_code=503, detail=f"Exception: {err}") from err
    return ModelResponse(prediction=prediction)


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={403: {"description": "Forbidden"}},
    name="health:get-data",
    dependencies=[Depends(token_auth_scheme)],  # api token authentication
)
async def health() -> HealthResponse:
    """
    **Health Check API**

    Check the health status of the API.

    This endpoint performs a health check to ensure that the API is running
    properly.

    **Output**
    - `HealthResponse`: The health status of the API.

    **Raises**
    - `HTTPException 503`: If the health check fails.
    """
    is_health = False
    try:
        with open(INPUT_EXAMPLE, "r", encoding="utf-8") as stream:
            example = json.load(stream)
            test_input = ModelInput(**example)
        test_point = test_input.get_df()
        get_prediction(test_point)
        is_health = True
        return HealthResponse(status=is_health)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable",
        ) from exc


@log_router.get(
    "/predict-logs",
    response_model=list[PredictLogEntry],
    responses={403: {"description": "Forbidden"}},
    name="predict-logs:get-data",
    dependencies=[Depends(token_auth_scheme)],  # api token authentication
)
async def get_log_entries(
    limit: int = Query(
        default=10,
        description="Number of log entries to retrieve",
    )
) -> list[PredictLogEntry]:
    """
    **Prediction Logs API**

    Get historical predict inputs and output entries.

    This endpoint retrieves the latest log entries from a JSON file and returns
    them as a list.

    **Input**
    - `limit` (int): The maximum number of log entries to retrieve
    (default: 10).

    **Output**
    - `list[PredictLogEntry]`: A list of log entries as dictionaries.
    """
    log_entries_data = read_log_entries(limit)
    log_entries = []

    for entry in log_entries_data:
        input_data_list = entry.get("input", [])
        if isinstance(input_data_list, list):
            input_list = []
            for input_data in input_data_list:
                if isinstance(input_data, dict):
                    input_list.append(ModelInput(**input_data))
        else:
            input_list = []

        result = entry.get("result")
        result = (
            float(result) if isinstance(result, (float, int, str)) else 0.0
        )  # Convert to float if possible

        date_str = entry.get("date", "")
        date = (
            datetime.datetime.fromisoformat(date_str)
            if isinstance(date_str, str)
            else datetime.datetime.now()
        )  # Parse date

        log_entry = PredictLogEntry(input=input_list, result=result, date=date)
        log_entries.append(log_entry)

    return log_entries
