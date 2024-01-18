"""
This module defines the data models used in the application. Namely the
model input features, model response (target) and a health check response.
"""

import pandas as pd
from pydantic import BaseModel


class ModelResponse(BaseModel):
    """Response model for the prediction endpoint."""

    prediction: float


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    status: bool


class ModelInput(BaseModel):
    """Input model for the prediction endpoint."""

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
