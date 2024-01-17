from typing import Callable

from fastapi import FastAPI


def preload_model():
    """
    In order to load model on memory to each worker
    """
    from app.services.predict import ModelHandlerScore
    from joblib import load

    ModelHandlerScore.get_model(load_wrapper=load)


def create_start_app_handler(app: FastAPI) -> Callable:
    def start_app() -> None:
        preload_model()

    return start_app
