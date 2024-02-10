"""
Module for handling model preloading and startup handling in FastAPI.
"""

import typing
from pickle import load

from fastapi import FastAPI

from app.services.predict import ModelHandlerScore


@typing.no_type_check
def preload_model():
    """Load the model into memory for each worker."""
    ModelHandlerScore.get_model(load_wrapper=load)


@typing.no_type_check
def create_start_app_handler(app: FastAPI):
    # pylint: disable=unused-argument
    """Create a startup handler function for FastAPI application."""

    def start_app() -> None:
        try:
            preload_model()
        except FileNotFoundError:
            pass  # Error logging is inside preload_model()

    return start_app
