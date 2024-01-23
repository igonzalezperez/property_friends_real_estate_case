"""
Configuration Module

This module provides configuration settings for the FastAPI application.
It includes settings related to debug mode, API version, project name, and
logging configuration.

Attributes:
    - API_PREFIX: The API route prefix.
    - VERSION: The API version.
    - DEBUG: Boolean flag indicating whether the application is in debug mode.
    - PRE_LOAD: Boolean flag indicating whether to pre-load data.
    - MAX_CONNECTIONS_COUNT: The maximum number of database connections.
    - MIN_CONNECTIONS_COUNT: The minimum number of database connections.
    - PROJECT_NAME: The name of the FastAPI project.

Logging Configuration:
    The module configures logging based on the DEBUG flag. It sets the logging
    level and configures loggers.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger
from starlette.config import Config

from app.core.logging import InterceptHandler

load_dotenv(find_dotenv())
config = Config(".env")

API_PREFIX = "/api"
VERSION = "0.1.0"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
PRE_LOAD: bool = config("PRE_LOAD", cast=bool, default=True)
MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)
PROJECT_NAME: str = config(
    "PROJECT_NAME",
    default="Property friends real estate case",
)

# logging configuration
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])

# Get the directory of the current script (the FastAPI app)
current_dir = os.path.dirname(__file__)

INPUT_EXAMPLE = str(
    Path(
        str(os.getenv("ML_DATA_DIR")),
        "examples",
        "example.json",
    )
)
