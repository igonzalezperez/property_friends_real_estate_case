import sys
import logging

from loguru import logger
from starlette.config import Config
from starlette.datastructures import Secret

from app.core.logging import InterceptHandler
import os
from pathlib import Path

config = Config(".env")

API_PREFIX = "/api"
VERSION = "0.1.0"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
PRE_LOAD: bool = config("PRE_LOAD", cast=bool, default=True)
MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)
CLIENT_API_KEY: Secret = config("CLIENT_API_KEY", cast=Secret, default="my_api_key")
PROJECT_NAME: str = config("PROJECT_NAME", default="Property friends real estate case")
MODEL_NAME = config("MODEL_NAME", default="model.joblib")

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
logger.info(f"MODEL_NAME: {MODEL_NAME}")
logger.info(f"ML_DIR {ML_DIR}")
logger.info(f"ML_MODELS_DIR {ML_MODELS_DIR}")
logger.info(f"ML_DATA_DIR {ML_DATA_DIR}")
logger.info(f"INPUT_EXAMPLE {INPUT_EXAMPLE}")
