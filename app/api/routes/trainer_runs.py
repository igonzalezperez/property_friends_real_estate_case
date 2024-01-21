"""
This module defines the API endpoints related to showing training model runs.
"""
from typing import Any

import mlflow
from fastapi import APIRouter, Depends

from app.core.middleware.api_key_middleware import token_auth_scheme
from app.models.train_runs import ModelRuns

router = APIRouter()


@router.get(
    "/model-runs",
    response_model=list[ModelRuns],
    name="get:model-runs",
    dependencies=[Depends(token_auth_scheme)],
)
def get_model_runs() -> Any:
    """
    Retrieves a list of MLFLOW model runs with details.

    This endpoint searches for model runs in MLflow with experiment ID '1' and
    returns the results as a list of `ModelRuns`.

    :return list[ModelRuns]: A list of model runs.
    """
    df = mlflow.search_runs("1")
    # Convert column names to python-like variables
    df.columns = [i.replace(".", "_").replace("-", "_") for i in df.columns]
    # Convert DataFrame to a list of dictionaries for JSON serialization
    return df.to_dict(orient="records")
