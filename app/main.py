from fastapi import FastAPI

from app.api.routes.api import router as api_router
from app.core.events import create_start_app_handler
from app.core.config import API_PREFIX, DEBUG, PROJECT_NAME, VERSION, PRE_LOAD


def get_application() -> FastAPI:
    application = FastAPI(title=PROJECT_NAME, debug=DEBUG, version=VERSION)
    application.include_router(api_router, prefix=API_PREFIX)
    if PRE_LOAD:
        application.add_event_handler("startup", create_start_app_handler(application))
    return application


app = get_application()
