"""
Configuration and setup for the FastAPI application.

This module handles configuration settings through environment variables,
logging setup, and paths for the FastAPI application.

Attributes:
    API_PREFIX (str): The API prefix for the application.
    VERSION (str): The version of the application.
    DEBUG (bool): A flag indicating whether the application is in debug mode.
    PRE_LOAD (bool): A flag indicating whether pre-loading is enabled.
    MAX_CONNECTIONS_COUNT (int): The maximum number of connections allowed.
    MIN_CONNECTIONS_COUNT (int): The minimum number of connections required.
    CLIENT_API_KEY (Secret): The client API key for authentication.
    PROJECT_NAME (str): The name of the FastAPI project.
    BASE_MODEL_NAME (str): The default name for the machine learning model.

    LOGGING_LEVEL (int): The logging level for the application.
    logger (Logger): The logger instance for the application.
    ML_DIR (str): The path to the 'ml' directory.
    ML_MODELS_DIR (str): The path to the 'models' directory within the 'ml'
    directory.
    ML_DATA_DIR (str): The path to the 'data' directory within the 'ml'
    directory.
    INPUT_EXAMPLE (str): The path to the example input data file.
"""
import logging
import os
import sys
from pathlib import Path

from loguru import logger
from starlette.config import Config
from starlette.datastructures import Secret

from app.core.logging import InterceptHandler

config = Config(".env")

API_PREFIX = "/api"
VERSION = "0.1.0"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
PRE_LOAD: bool = config("PRE_LOAD", cast=bool, default=True)
MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)
CLIENT_API_KEY: Secret = config(
    "CLIENT_API_KEY", cast=Secret, default=Secret("my_api_key")
)
PROJECT_NAME: str = config("PROJECT_NAME", default="Property friends real estate case")
BASE_MODEL_NAME: str = config("BASE_MODEL_NAME", default="model.joblib")
MLFLOW_TRACKING_URI: str = config(
    "MLFLOW_TRACKING_URI", default="sqlite:///ml/mlflow/mlflow.db"
)
MLFLOW_ARTIFACT_ROOT: str = config("MLFLOW_ARTIFACT_ROOT", default="/ml/models/mlflow")


# logging configuration
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])

# Get the directory of the current script (the FastAPI app)
current_dir = os.path.dirname(__file__)

# Construct the path to the 'ml' directory
ML_DIR = os.path.abspath(os.path.join(current_dir, "../../ml"))

# Access files and folders within the 'ml' directory
ML_MODELS_DIR = os.path.join(ML_DIR, "models")
ML_DATA_DIR = os.path.join(ML_DIR, "data")

INPUT_EXAMPLE = str(Path(ML_DATA_DIR, "examples", "example.json"))

logger.info(f"DEBUG: {DEBUG}")
logger.info(f"PRE_LOAD: {PRE_LOAD}")
logger.info(f"MAX_CONNECTIONS_COUNT: {MAX_CONNECTIONS_COUNT}")
logger.info(f"MIN_CONNECTIONS_COUNT: {MIN_CONNECTIONS_COUNT}")
logger.info(f"CLIENT_API_KEY: {CLIENT_API_KEY}")
logger.info(f"PROJECT_NAME: {PROJECT_NAME}")
logger.info(f"BASE_MODEL_NAME: {BASE_MODEL_NAME}")
logger.info(f"ML_DIR: {ML_DIR}")
logger.info(f"ML_MODELS_DIR: {ML_MODELS_DIR}")
logger.info(f"ML_DATA_DIR: {ML_DATA_DIR}")
logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
logger.info(f"MLFLOW_ARTIFACT_ROOT: {MLFLOW_ARTIFACT_ROOT}")
logger.info(f"INPUT_EXAMPLE: {INPUT_EXAMPLE}")
