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
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import dump, load
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor


def pipeline(
    input_data_dir: str,
    input_model_dir: str,
    output_model_dir: str,
) -> None:
    """
    Process data and train a Gradient Boosting Regressor model.

    :param str input_data_dir: The directory containing the cleaned input data
    :param str input_model_dir: The directory containing the saved
    preprocessing pipeline and model
    :param str output_model_dir: The directory where the trained model will be saved
    """
    logger.info("Start building features.")
    data_path = Path(input_data_dir, "train_data.csv")
    preproc_path = Path(input_model_dir, "preproc_pipeline.joblib")
    model_path = Path(output_model_dir, "preproc_pipeline.joblib")

    logger.info(f"Reading data from: {data_path}")
    logger.info(f"Reading preprocessing pipeline from: {preproc_path}")
    data = pd.read_csv(data_path)
    preproc_pipeline = load(preproc_path)

    target = "price"
    feature_cols = [col for col in data.columns if col != target]
    input_features = preproc_pipeline.transform(data[feature_cols])
    model = GradientBoostingRegressor(
        **{
            "learning_rate": 0.01,
            "n_estimators": 300,
            "max_depth": 5,
            "loss": "absolute_error",
        }
    )
    logger.info(f"Training model: {model.__class__} with parameters: {model.__dict__}")
    model.fit(input_features, data[target])
    logger.info(f"Saving model to {model_path}")
    dump(model, model_path)


@click.command()
@click.argument(
    "input_data_dir", default="ml/data/interim", type=click.Path(exists=True)
)
@click.argument("input_model_dir", default="ml/models", type=click.Path(exists=True))
@click.argument("output_model_dir", default="ml/models", type=click.Path())
def main(
    input_data_dir: str,
    input_model_dir: str,
    output_model_dir: str,
) -> None:
    """
    Run data processing scripts to preprocess cleaned data from (../interim)
    and train a machine learning model.
    The trained model is saved in (../models).
    """
    logger.info(
        f"Read from {input_data_dir}, {input_model_dir}, write to {output_model_dir}."
    )
    pipeline(input_data_dir, input_model_dir, output_model_dir)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-parameter
    main()
