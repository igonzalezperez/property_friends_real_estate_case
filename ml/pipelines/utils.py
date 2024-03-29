"""
Utilities for ML pipelines
"""
import os
from pathlib import Path
from typing import Any, Callable

import mlflow
import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from numpy import float64, sqrt
from numpy.typing import NDArray
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.exc import NoSuchTableError, OperationalError, SQLAlchemyError

load_dotenv(find_dotenv())
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": "db",
    "port": "5432",
}

DB_CONN_STR = f'postgresql://{DB_PARAMS["user"]}:{DB_PARAMS["password"]}@{DB_PARAMS["host"]}:{DB_PARAMS["port"]}/{DB_PARAMS["dbname"]}'

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

    # Dynamically import all necessary column transformers
    for k, v in config["model_pipeline"]["preprocessor"]["transformers"].items():
        params[k + "_transform"] = import_from_path(v)

    model_class_path = config["model_pipeline"]["model"]["type"]
    model_params = config["model_pipeline"]["model"]["parameters"]

    metrics_paths = config["model_pipeline"]["metrics"]

    model = import_from_path(model_class_path)

    metrics = [import_from_path(i) for i in metrics_paths]

    params["num_cols"] = num_cols
    params["cat_cols"] = cat_cols
    params["feature_cols"] = feature_cols
    params["target_col"] = target_col
    params["model_params"] = model_params
    params["model"] = model
    params["model_class_path"] = model_class_path
    params["metrics"] = metrics
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


def get_table_as_df(db_conn_str: str, table_name: str) -> pd.DataFrame:
    """
    Retrieves data from a database table and returns it as a DataFrame.

    :param str db_conn_str: DB connection string.
    :param str table_name: The name of the table to retrieve data from.
    :return pd.DataFrame: A DataFrame containing the data from the specified
    table.
    :raises sqlalchemy.exc.SQLAlchemyError: If there is an error with the
    database connection.
    """
    engine = create_engine(db_conn_str)
    try:
        df = pd.read_sql(table_name, con=engine)
        return df
    except SQLAlchemyError as e:
        logger.error(f"Database connection error: {e}")
        raise


def reset_raw_data_table(db_conn_str: str) -> None:
    """
    Deletes all records from the 'raw_data' table in the database.
    Useful for testing purposes.

    :param db_conn_str: A string representing the database connection string.
                        The format of this string should be appropriate for
                        the specific type of database you are connecting to.
                        For example, for a PostgreSQL database, it should
                        be in the format:
                        'postgresql://username:password@host:port/dbname'
    """
    # Create an engine
    engine = create_engine(db_conn_str)

    # SQL command to delete all records from the table
    sql = text("DELETE FROM raw_data")

    # Connect to the database and execute the command
    with engine.connect() as conn:
        conn.execute(sql)
        # Commit the transaction if needed
        conn.commit()

    logger.info("All records from 'raw_data' table have been deleted.")


def drop_raw_data_table(db_conn_str: str) -> None:
    """
    Drop raw_data table. Useful for testing purposes
    Useful for testing purposes.

    :param db_conn_str: A string representing the database connection string.
                        The format of this string should be appropriate for
                        the specific type of database you are connecting to.
                        For example, for a PostgreSQL database, it should
                        be in the format:
                        'postgresql://username:password@host:port/dbname'
    """
    try:
        # Create an engine
        engine = create_engine(db_conn_str)

        # Create a MetaData instance
        metadata = MetaData()

        # Reflect the table from the database
        table = Table("raw_data", metadata, autoload_with=engine)

        # Drop the table
        table.drop(engine, checkfirst=True)
        logger.info("Dropped table `raw_data`")
    except NoSuchTableError:
        logger.error("Table 'raw_data' does not exist and cannot be dropped.")
    except OperationalError as e:
        logger.error(
            f"""Operation error: {e}. Please check your database connection
              string and ensure the database server is accessible."""
        )
