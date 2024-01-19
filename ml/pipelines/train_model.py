"""
Data Processing Pipeline

This script defines a data processing pipeline that transforms cleaned data
and builds a machine learning model. It reads the cleaned data from a CSV file
in 'input_data_dir', applies preprocessing using a saved pipeline from
'input_model_dir', and then trains a Gradient Boosting Regressor model.
The trained model is saved in 'output_model_dir'.

Usage:
    To run this script, use the following command:
    ```
    python process_data_and_train_model.py [INPUT_DATA_DIR] [INPUT_MODEL_DIR]
    [OUTPUT_MODEL_DIR]
    ```

Arguments:
    INPUT_DATA_DIR (str): The directory containing the cleaned input data
    (default: "ml/data/interim").
    INPUT_MODEL_DIR (str): The directory containing the saved preprocessing
    pipeline and model (default: "ml/models").
    OUTPUT_MODEL_DIR (str): The directory where the trained model will be saved
    (default: "ml/models").

Example:
    ```
    python process_data_and_train_model.py ml/data/interim ml/models ml/models
    ```
"""
from pathlib import Path

import click
import mlflow
import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv
from joblib import dump, load
from loguru import logger
from sklearn.pipeline import Pipeline

from app.core.config import (
    BASE_MODEL_NAME,
    ML_DATA_DIR,
    ML_DIR,
    ML_MODELS_DIR,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
)

with open(
    Path(ML_DIR, "pipelines", "pipeline_config.yml"), "r", encoding="utf-8"
) as file:
    config = yaml.safe_load(file)

NUM_COLS = config["data_catalog"]["columns"]["features"]["numerical"]
CAT_COLS = config["data_catalog"]["columns"]["features"]["categorical"]
FEATURE_COLS = [*CAT_COLS, *NUM_COLS]
TARGET_COL = config["data_catalog"]["columns"]["target"]

MODEL_CLASS_PATH = config["model_pipeline"]["model"]["type"]
MODEL_PARAMS = config["model_pipeline"]["model"]["parameters"]

MODEL_MODULE, MODEL_CLASS = MODEL_CLASS_PATH.rsplit(".", 1)
MODEL_ = __import__(MODEL_MODULE, fromlist=[MODEL_CLASS])
MODEL = getattr(MODEL_, MODEL_CLASS)


def pipeline(
    input_dir: str,
    input_file: str,
    input_pipeline_dir: str,
    input_pipeline_file: str,
    output_path: str,
) -> None:
    """
    Build a data preprocessing pipeline to transform input data.

    :param Union[str, Path] input_dir: Input data directory
    :param Union[str, Path] input_file: Input file name
    :param Union[str, Path] output_model_path: Output model file path
    """
    logger.info("Start building features.")
    input_path = Path(input_dir, input_file)
    input_pipeline_path = Path(input_pipeline_dir, input_pipeline_file)

    logger.info(f"Reading preprocessed data from: {input_path}")
    data = pd.read_csv(input_path)
    logger.info(f"Reading processing pipeline data from: {input_pipeline_path}")
    preproc_pipeline = load(input_pipeline_path)

    logger.info(f"Model Features: {FEATURE_COLS}")
    logger.info(f"Model Target: {TARGET_COL}")

    logger.info(f"Model Type: {MODEL}")
    logger.info(f"Model Params: {MODEL_PARAMS}")

    model = MODEL(**MODEL_PARAMS)

    logger.info(f"Training model: {model.__class__} with parameters: {model.__dict__}")
    model_pipe = Pipeline([("preprocessor", preproc_pipeline), ("model", model)])
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiments = mlflow.search_experiments(filter_string="name = 'TrainModel'")
    if not experiments:
        exp_id = mlflow.create_experiment(
            "TrainModel", artifact_location=MLFLOW_ARTIFACT_ROOT + "/1"
        )
    else:
        exp_id = experiments[-1].experiment_id
    print(exp_id)
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.set_tag("model_name", MODEL_CLASS_PATH)
        mlflow.log_params(MODEL_PARAMS)
        model_pipe.fit(data[FEATURE_COLS], data[TARGET_COL])
        logger.info(f"Saving model to {output_path}")
        dump(model_pipe, output_path)
        # with open(output_path, "wb", encoding="utf-8") as stream:
        #     pickle.dump(model_pipe, stream)
        mlflow.sklearn.log_model(
            sk_model=model_pipe,
            artifact_path=Path(output_path).stem,
        )

        logger.debug(MLFLOW_ARTIFACT_ROOT)
        logger.debug(mlflow.get_artifact_uri())
    logger.success("Successfully ran TrainModel")


@click.command()
@click.option(
    "--input-file-dir",
    "input_dir",
    default=Path(ML_DATA_DIR, "interim"),
    type=click.Path(exists=True),
)
@click.option(
    "--input-file",
    "input_file",
    default="preprocessed_train.csv",
)
@click.option(
    "--input-pipeline-dir",
    "input_pipeline_dir",
    default=ML_MODELS_DIR,
    type=click.Path(exists=True),
)
@click.option(
    "--input-pipeline-file",
    "input_pipeline_file",
    default="preproc_pipeline.joblib",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=ML_MODELS_DIR,
    type=click.Path(exists=True),
)
def main(
    input_dir: str,
    input_file: str,
    input_pipeline_dir: str,
    input_pipeline_file: str,
    output_dir: str,
) -> None:
    """
    Run data processing scripts to preprocess cleaned data from (../interim)
    and train a machine learning model.
    The trained model is saved in (../models).
    """
    if not Path(input_dir, input_file).exists():
        raise FileNotFoundError(f"File '{Path(input_dir, input_file)}' doesn't exist")
    if not Path(input_pipeline_dir, input_pipeline_file).exists():
        raise FileNotFoundError(
            f"File '{Path(input_pipeline_dir, input_pipeline_file)}' doesn't exist"
        )
    output_path = str(Path(output_dir, BASE_MODEL_NAME))
    pipeline(
        input_dir,
        input_file,
        input_pipeline_dir,
        input_pipeline_file,
        output_path,
    )


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-parameter
    main()
