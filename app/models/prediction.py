import pandas as pd

from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: bool


class MachineLearningDataInput(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float

    def get_df(self):
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
