import json
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.core.middleware.api_key_middleware import token_auth_scheme
from app.core.logger_config import logger
from app.core.monitoring import save_to_text
from app.core.config import INPUT_EXAMPLE
from app.services.predict import ModelHandlerScore as model
from app.models.prediction import (
    HealthResponse,
    ModelResponse,
    ModelInput,
)

router = APIRouter()


def get_prediction(data_point: pd.DataFrame) -> float:
    return model.predict(data_point, load_wrapper=joblib.load, method="predict")


@router.post(
    "/predict",
    response_model=ModelResponse,
    name="predict:get-data",
    dependencies=[Depends(token_auth_scheme)],
)
async def predict(data_input: ModelInput, background_tasks: BackgroundTasks):
    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    try:
        data_point = data_input.get_df()
        prediction = get_prediction(data_point)
        background_tasks.add_task(save_to_text, input=data_point, result=prediction)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return ModelResponse(prediction=prediction)


@router.get(
    "/health",
    response_model=HealthResponse,
    name="health:get-data",
    dependencies=[Depends(token_auth_scheme)],
)
async def health():
    is_health = False
    try:
        test_input = ModelInput(**json.loads(open(INPUT_EXAMPLE, "r").read()))
        test_point = test_input.get_df()
        get_prediction(test_point)
        is_health = True
        return HealthResponse(status=is_health)
    except Exception:
        raise HTTPException(status_code=404, detail="Unhealthy")
