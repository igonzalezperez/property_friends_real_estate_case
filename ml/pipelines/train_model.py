"""
Model Training Pipeline

This script trains a predictive Model from data in ../interim and a previously
fitted pipeline at ../models to get input features. The trained model is logged
using MLFlow and is saved in ml/models/trained_model. If the TEST_PREDICT env
variable is set to True it also logs the accuracy on test set.

Usage:
    To run this script, use the following command:
    ```
    (poetry run) python ml/pipelines/make_dataset.py [INPUT_DIR]
    [INPUT_FILE] [INPUT_TEST_FILE] [INPUT_PIPELINE_DIR] [INPUT_PIPELINE_FILE]
    ```
"""
import os
import shutil
from pathlib import Path

import click
import mlflow
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import load
from loguru import logger
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from ml.pipelines.utils import get_pipeline_config, log_metrics
from starlette.config import Config

load_dotenv(find_dotenv())

config = Config(".env")
TEST_PREDICT: bool = config("TEST_PREDICT", cast=bool, default=True)


def pipeline(
    input_dir: str,
    input_file: str,
    input_test_file: str,
    input_pipeline_dir: str,
    input_pipeline_file: str,
) -> None:
    """
    Build a ML training pipeline and save fitted model.

    :param str input_dir: Input data directory
    :param str input_file: Input file name
    :param str input_test_file: Input test file name
    :param str input_pipeline_dir: Input pipeline model directory
    :param str input_pipeline_file: Input pipeline model file
    """
    # Check if input paths exists
    valid_test = True
    if not Path(input_dir, input_file).exists():
        raise FileNotFoundError(f"File '{Path(input_dir, input_file)}' doesn't exist")
    if not Path(input_dir, input_file).exists():
        logger.warning(f"File '{Path(input_dir, input_test_file)}' doesn't exist")
        valid_test = False
    if not Path(input_pipeline_dir, input_pipeline_file).exists():
        raise FileNotFoundError(
            f"File '{Path(input_pipeline_dir, input_pipeline_file)}' doesn't exist"
        )

    # Load pipeline params
    logger.info("Load data pipeline config.")
    params = get_pipeline_config()
    logger.info("Start building features.")

    # Load input data and pipeline
    input_path = Path(input_dir, input_file)
    input_test_path = Path(input_dir, input_test_file)
    input_pipeline_path = Path(input_pipeline_dir, input_pipeline_file)

    logger.info(f"Reading preprocessed data from: {input_path}")
    data = pd.read_csv(input_path)
    logger.info(f"Reading processing pipeline data from: {input_pipeline_path}")
    preproc_pipeline = load(input_pipeline_path)

    logger.info(f"Model Features: {params['feature_cols']}")
    logger.info(f"Model Target: {params['target_col']}")

    logger.info(f"Model Type: {params['model']}")
    logger.info(f"Model Params: {params['model_params']}")

    # Initialize model
    model = params["model"](**params["model_params"])

    logger.info(f"Training model: {model.__class__} with parameters: {model.__dict__}")
    model_pipe = Pipeline([("preprocessor", preproc_pipeline), ("model", model)])

    # Use existing MLFlow experiment or create a new one if there isn't any
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    experiments = mlflow.search_experiments(filter_string="name = 'TrainModel'")
    if not experiments:
        artifact_location = str(Path(str(os.getenv("MLFLOW_ARTIFACT_ROOT")), "1"))
        exp_id = mlflow.create_experiment(
            "TrainModel", artifact_location=artifact_location
        )
    else:
        exp_id = experiments[-1].experiment_id

    # Training
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_params(params["model_params"])
        data[params["target_col"]] = data[params["target_col"]].astype("float")
        model_pipe.fit(data[params["feature_cols"]], data[params["target_col"]])
        signature = infer_signature(
            data[params["feature_cols"]],
            data[params["target_col"]],
            params["model_params"],
        )
        mlflow.sklearn.log_model(
            sk_model=model_pipe,
            artifact_path="model",
            signature=signature,
        )
        model_path = Path(
            str(os.getenv("ML_MODELS_DIR")),
            "trained_model",
        )
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        mlflow.sklearn.save_model(
            sk_model=model_pipe,
            path=model_path,
        )

        # Train metrics
        preds = model_pipe.predict(
            data[params["feature_cols"]],
        )
        log_metrics(
            params["metrics"],
            data[params["target_col"]].values,
            preds,
            "train",
        )
        # Test metrics
        if TEST_PREDICT and valid_test:
            logger.info(f"Reading preprocessed TEST data from: {input_path}")
            data_test = pd.read_csv(input_test_path)
            data_test[params["target_col"]] = data_test[params["target_col"]].astype(
                "float"
            )
            test_preds = model_pipe.predict(
                data_test[params["feature_cols"]],
            )
            log_metrics(
                params["metrics"],
                data_test[params["target_col"]].values,
                test_preds,
                "test",
            )
        logger.debug(os.getenv("MLFLOW_ARTIFACT_ROOT"))
        logger.debug(mlflow.get_artifact_uri())
    logger.success("Successfully ran train_model")


@click.command()
@click.option(
    "--input-dir",
    "input_dir",
    default=Path(str(os.getenv("ML_DATA_DIR")), "interim"),
    type=click.Path(exists=True),
)
@click.option(
    "--input-file",
    "input_file",
    default=f"{str(os.getenv('INTERIM_DATA_PREFIX'))}_{str(os.getenv('TRAIN_FILE_NAME'))}",
)
@click.option(
    "--input-test-file",
    "input_test_file",
    default=f"{str(os.getenv('INTERIM_DATA_PREFIX'))}_{str(os.getenv('TEST_FILE_NAME'))}",
)
@click.option(
    "--input-pipeline-dir",
    "input_pipeline_dir",
    default=os.getenv("ML_MODELS_DIR"),
    type=click.Path(exists=True),
)
@click.option(
    "--input-pipeline-file",
    "input_pipeline_file",
    default=os.getenv("PREPROC_FILE_NAME"),
)
def run(
    input_dir: str,
    input_file: str,
    input_test_file: str,
    input_pipeline_dir: str,
    input_pipeline_file: str,
) -> None:
    """
    Run data processing scripts to preprocess cleaned data from (../interim)
    and train a machine learning model.
    The trained model is saved in (../models).
    """
    pipeline(
        input_dir,
        input_file,
        input_test_file,
        input_pipeline_dir,
        input_pipeline_file,
    )


if __name__ == "__main__":
    # pylint: disable = no-value-for-parameter
    run()
