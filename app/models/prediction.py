"""
This module defines the data models used in the application. Namely the
model input features, model response (target) and a health check response.
"""

import pandas as pd
from pydantic import BaseModel
import datetime


class ModelResponse(BaseModel):
    """
    Response model for the prediction endpoint.

    Fields:
    - `prediction` (float): The prediction result.
    """

    prediction: float


class HealthResponse(BaseModel):
    """
    Response model for the health endpoint.

    Fields:
    - `status` (bool): The health status.
    """

    status: bool


class ModelInput(BaseModel):
    """
    Input model for the real estat price prediction endpoint.

    Fields:
    - `type` (str): The type of real estate.
    - `sector` (str): The sector.
    - `net_usable_area` (float): The net usable area.
    - `net_area` (float): The net area.
    - `n_rooms` (float): The number of rooms.
    - `n_bathroom` (float): The number of bathrooms.
    - `latitude` (float): The latitude.
    - `longitude` (float): The longitude.

    Methods:
    - `get_df() -> pd.DataFrame`: Converts the input data to a Pandas DataFrame
    """

    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float

    def get_df(self) -> pd.DataFrame:
        """Converts the input data to a Pandas DataFrame."""
        data = {
            "type": [self.type],
            "sector": [self.sector],
            "net_usable_area": [self.net_usable_area],
            "net_area": [self.net_area],
            "n_rooms": [self.n_rooms],
            "n_bathroom": [self.n_bathroom],
            "latitude": [self.latitude],
            "longitude": [self.longitude],
        }
        return pd.DataFrame(data)


class PredictLogEntry(BaseModel):
    """
    Log entry model for the log API.

    Fields:
    - `input` (list[ModelInput]): Thelist of input data for the model
    predictions.
    - `result` (ModelResponse): The output result of the model prediction.
    - `date` (datetime.datetime): The date and time when the log entry was
    created.
    """

    input: list[ModelInput]
    result: float
    date: datetime.datetime
