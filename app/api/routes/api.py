"""
This module defines the APIRouter for the main application.

It includes the routes defined in the 'predictor' and 'trainer_runs' modules
and sets up the routing for the API.
"""

from fastapi import APIRouter

from app.api.routes import predictor, trainer_runs

router = APIRouter()
router.include_router(predictor.router, tags=["predictor"], prefix="/v1")
router.include_router(
    predictor.log_router,
    tags=["predict-logs"],
    prefix="/v1",
)
router.include_router(
    trainer_runs.router,
    tags=["train-runs-info"],
    prefix="/v1",
)
