"""
This module defines the API endpoints related to showing training model runs.

It includes the following routes:
- `/model-runs`: Gets the logs of all model training runs registered by
MLFlow
"""
from pathlib import Path
from typing import Any

import mlflow
from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, Depends
from starlette.config import Config

from app.core.middleware.api_key_middleware import token_auth_scheme
from app.models.train_runs import ModelRuns

load_dotenv(find_dotenv())

config = Config(".env")
ML_MODELS_DIR: str = config(
    "ML_MODELS_DIR",
    cast=str,
    default="ml/models",
)
router = APIRouter()


@router.get(
    "/model-runs",
    response_model=list[ModelRuns],
    responses={403: {"description": "Forbidden"}},
    name="get:model-runs",
    dependencies=[Depends(token_auth_scheme)],
)
def get_model_runs() -> Any:
    """
    **Model Runs Logs API**

    Retrieves a list of MLFLOW model runs with details.

    This endpoint searches for model runs in MLflow with experiment ID '1'
    (default) and returns the results as a list of `ModelRuns`.

    **Output**
    - `list[ModelRuns]`: A list of model runs from MLFlow. It includes metadata
    and metrics
    """
    if not Path(ML_MODELS_DIR, "mlflow", "mlflow.db").exists():
        return []
    df = mlflow.search_runs("1")
    # Convert column names to python-like variables
    df.columns = [i.replace(".", "_").replace("-", "_") for i in df.columns]
    # Convert DataFrame to a list of dictionaries for JSON serialization
    return df.to_dict(orient="records")
