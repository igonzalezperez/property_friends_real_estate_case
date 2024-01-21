"""
Utilities for ML pipelines
"""
import os
from pathlib import Path
from typing import Any, Callable

import mlflow
import yaml
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from numpy import float64, sqrt
from numpy.typing import NDArray

load_dotenv(find_dotenv())

# type hint for metric functions
MetricFunction = Callable[[NDArray[float64], NDArray[float64]], float]


def import_from_path(path: str) -> Any:
    """
    Dynamically imports a class or function from a given module path

    :param str path: The full path to the class or function to be imported,
    formatted as 'module.submodule.ClassOrFunction'
    :return Any: The imported class or function object
    """
    module_name, module_class = path.rsplit(".", 1)
    module_import = __import__(module_name, fromlist=[module_class])
    module = getattr(module_import, module_class)
    return module


def get_pipeline_config() -> dict[str, Any]:
    """
    Reads and parses the pipeline configuration file, 'pipeline_config.yml'
    located in the 'pipelines' directory within the ML directory defined
    by the 'ML_DIR' environment variable. Constructs and returns a dictionary
    of parameters and objects necessary for building and running the ML
    pipeline.

    :return dict[Any]: A dictionary containing configuration parameters and
    dynamically imported objects based on the pipeline configuration.
    """
    params = {}
    with open(
        Path(
            os.getenv("ML_DIR", default="ml"),
            "pipelines",
            "pipeline_config.yml",
        ),
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    num_cols = config["data_catalog"]["columns"]["features"]["numerical"]
    cat_cols = config["data_catalog"]["columns"]["features"]["categorical"]
    feature_cols = [*cat_cols, *num_cols]
    target_col = config["data_catalog"]["columns"]["target"]

    cat_transform_path = config["model_pipeline"]["preprocessor"]["transformers"][
        "categorical"
    ]
    num_transform_path = config["model_pipeline"]["preprocessor"]["transformers"][
        "numerical"
    ]
    model_class_path = config["model_pipeline"]["model"]["type"]
    model_params = config["model_pipeline"]["model"]["parameters"]

    metrics_paths = config["model_pipeline"]["metrics"]

    cat_transform = import_from_path(cat_transform_path)
    num_transform = import_from_path(num_transform_path)
    model = import_from_path(model_class_path)

    metrics = [import_from_path(i) for i in metrics_paths]

    params = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "cat_transform": cat_transform,
        "num_transform": num_transform,
        "model_params": model_params,
        "model": model,
        "model_class_path": model_class_path,
        "metrics": metrics,
    }
    return params


def log_metrics(
    metrics: list[MetricFunction],
    target: NDArray[float64],
    preds: NDArray[float64],
    suffix: str = "train",
) -> None:
    """
    Logs given metrics to MLflow. Converts mean squared error to root mean
    squared error for better interpretation. Appends a specified suffix to
    metric names for differentiation.

    :param List[MetricFunction] metrics: A list of metric functions to be
    applied.
    :param NDArray[float] target: The true target values.
    :param NDArray[float] preds: The predicted values.
    :param str suffix: A suffix to be appended to each metric name, defaults to
    "train".
    """
    metrics_dict = {}
    for metric_func in metrics:
        metric_name = metric_func.__name__
        metric_value = metric_func(
            target,
            preds,
        )
        if metric_name == "mean_squared_error":
            metric_name = "root_" + metric_name
            metric_value = sqrt(metric_value)
        metric_name += f"_{suffix}"
        metrics_dict[metric_name] = metric_value
        logger.info(f"{metric_name} = {metric_value}")
    mlflow.log_metrics(metrics_dict)
