"""
This module defines the data model used for MLFlow experiments created during
model training.
"""

import datetime
from typing import Optional

from pydantic import BaseModel


class ModelRuns(BaseModel):
    """
    Response model for MLFlow run training records

    Fields:
    - `run_id` (str): The unique identifier for the model run.
    - `experiment_id` (str): The identifier for the associated experiment.
    - `status` (str): The status of the model run (e.g., "FINISHED").
    - `artifact_uri` (str): The URI of the model's artifacts.
    - `start_time` (datetime.datetime): The timestamp when the model run
    started.
    - `end_time` (datetime.datetime): The timestamp when the model run ended.
    - `metrics_mean_absolute_error_train` (Optional[float]): Optional mean
    absolute error metric.
    - `metrics_mean_absolute_percentage_error_test` (Optional[float]): Optional
    mean absolute percentage error metric.
    - `metrics_root_mean_squared_error_train` (Optional[float]): Optional root
    mean squared error metric.
    - `metrics_mean_absolute_error_train` (Optional[float]): Optional mean
    absolute error metric.
    - `metrics_mean_absolute_percentage_error_test` (Optional[float]): Optional
    mean absolute percentage error metric.
    - `metrics_root_mean_squared_error_test` (Optional[float]): Optional root
    mean squared error metric.
    - `params_n_estimators` (str): The number of estimators used in the model.
    - `params_random_state` (str): The random state used in the model.
    - `params_loss` (str): The loss function used in the model.
    - `params_max_depth` (str): The maximum depth of the model.
    - `params_learning_rate` (str): The learning rate used in the model.
    - `tags_mlflow_runName` (str): The name associated with the MLflow run.
    - `tags_mlflow_source_name` (str): The source name of the MLflow run.
    - `tags_mlflow_user` (str): The user associated with the MLflow run.
    - `tags_mlflow_source_type` (str): The source type of the MLflow run.
    - `tags_mlflow_log_model_history` (str): A history of MLflow log model tags
    """

    run_id: str
    experiment_id: str
    status: str
    artifact_uri: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    metrics_mean_absolute_error_train: Optional[float]
    metrics_mean_absolute_percentage_error_train: Optional[float]
    metrics_root_mean_squared_error_train: Optional[float]
    metrics_mean_absolute_error_test: Optional[float]
    metrics_mean_absolute_percentage_error_test: Optional[float]
    metrics_root_mean_squared_error_test: Optional[float]
    params_n_estimators: str
    params_random_state: str
    params_loss: str
    params_max_depth: str
    params_learning_rate: str
    tags_mlflow_runName: str
    tags_mlflow_source_name: str
    tags_mlflow_user: str
    tags_mlflow_source_type: str
    tags_mlflow_log_model_history: str
