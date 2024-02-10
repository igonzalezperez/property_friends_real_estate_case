"""
Main FastAPI Application Configuration
"""

from fastapi import FastAPI

from app.api.routes.api import router as api_router
from app.core.config import API_PREFIX, DEBUG, PRE_LOAD, PROJECT_NAME, VERSION
from app.core.events import create_start_app_handler
from app.core.middleware.api_key_auth import ApiKeyManager

# Create an api key for root user on startup for testing purposes
ApiKeyManager().add_api_key("root")


def get_application() -> FastAPI:
    """
    Get the FastAPI application instance with configured settings. If env
    variable PRE_LOAD is set to True, it loads the ML model on app startup,
    instead of later when the API endpoint is called.

    :return FastAPI: The configured FastAPI application.
    """
    application = FastAPI(title=PROJECT_NAME, debug=DEBUG, version=VERSION)
    application.include_router(api_router, prefix=API_PREFIX)
    if PRE_LOAD:
        application.add_event_handler(
            "startup",
            create_start_app_handler(application),
        )
    return application


app = get_application()
