"""
This module defines the API routes and functions for making predictions and
checking the health of the model.

It includes the following routes:
- `/predict`: Takes input data, makes predictions, and returns the result.
- `/health`: Checks the health of the model by performing a test prediction.
"""
import json
import typing

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from numpy.typing import NDArray

from app.core.config import INPUT_EXAMPLE
from app.core.middleware.api_key_middleware import token_auth_scheme
from app.core.monitoring import save_to_json
from app.models.prediction import HealthResponse, ModelInput, ModelResponse
from app.services.predict import ModelHandlerScore as model

router = APIRouter()


@typing.no_type_check
def get_prediction(data_point: pd.DataFrame) -> NDArray[np.float64]:
    """
    Get predictions for the input data point. It performs a .predict() on
    a loaded model artifact, like an sklearn model or pipeline.

    :param pd.DataFrame data_point: The input data point for prediction.
    :return NDArray[np.float64]: Predicted value as an array of type
    np.float64.
    """
    return model.predict(data_point, load_wrapper=joblib.load, method="predict")


@router.post(
    "/predict",
    response_model=ModelResponse,
    name="predict:get-data",
    dependencies=[Depends(token_auth_scheme)],  # api token authentication
)
async def predict(
    data_input: ModelInput, background_tasks: BackgroundTasks
) -> ModelResponse:
    """Get predictions for the input data.

    This endpoint receives a ModelInput object and performs predictions on it.
    The payload and response fields are defined by ModelInput and ModelResponse

    :param ModelInput data_input: The input data point for prediction.
    :param BackgroundTasks background_tasks: A background task for saving
    prediction results.
    :raises HTTPException: If the 'data_input' argument is invalid.
    :raises HTTPException: If an exception occurs during prediction.
    :return ModelResponse: The prediction results.
    """
    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    try:
        data_point = data_input.get_df()
        prediction = get_prediction(data_point)
        background_tasks.add_task(
            save_to_json,
            data_input=data_point,
            result=prediction,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}") from err
    return ModelResponse(prediction=prediction)


@router.get(
    "/health",
    response_model=HealthResponse,
    name="health:get-data",
    dependencies=[Depends(token_auth_scheme)],  # api token authentication
)
async def health() -> HealthResponse:
    """
    Check the health status of the API.

    This endpoint performs a health check to ensure that the API is running
    properly. It runs a prediction on example data.

    :raises HTTPException: If the health check fails.
    :return HealthResponse: The health status of the API.
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
        raise HTTPException(status_code=404, detail="Unhealthy") from exc
