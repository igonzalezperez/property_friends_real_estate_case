"""
Data Processing Pipeline

This script defines a data processing pipeline using scikit-learn's ColumnTransformer and
category_encoders' TargetEncoder to preprocess and transform input data for training.

Usage:
    To run this script, use the following command:
    ```
    python preprocess.py [INPUT_DATA_DIR] [OUTPUT_MODEL_DIR]
    ```

Arguments:
    INPUT_DATA_DIR (str): The directory containing the input data
    (default: "ml/data/interim").
    OUTPUT_MODEL_DIR (str): The directory where the preprocessing pipeline
    model will be saved (default: "ml/models").

Example:
    ```
    python preprocess.py ml/data/interim ml/models
    ```
"""
from pathlib import Path

import click
import pandas as pd
from category_encoders import TargetEncoder
from dotenv import find_dotenv, load_dotenv
from joblib import dump
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def pipeline(input_data_dir: str, output_model_dir: str) -> None:
    """
    Build a data preprocessing pipeline to transform input data.

    :param str input_data_dir:  The directory containing the input data.
    :param str output_model_dir: The directory where the preprocessing pipeline
    model will be saved.
    """
    logger.info("Start building features.")
    data_path = Path(input_data_dir, "train_data.csv")
    model_path = Path(output_model_dir, "preproc_pipeline.joblib")

    logger.info(f"Reading data from: {data_path}")
    data = pd.read_csv(data_path)

    categorical_cols = ["type", "sector"]
    target = "price"
    feature_cols = [col for col in data.columns if col != target]
    logger.info(f"Feature cols: {feature_cols}")
    logger.info(f"Target col: {target}")

    categorical_transformer = TargetEncoder()

    preprocessor = ColumnTransformer(
        transformers=[("categorical", categorical_transformer, categorical_cols)]
    )

    steps = [
        ("preprocessor", preprocessor),
    ]
    preproc_pipeline = Pipeline(steps).fit(data[feature_cols], data[target])
    logger.info(f"Saving preprocessing pipeline to: {model_path}")
    dump(preproc_pipeline, Path(output_model_dir, "preproc_pipeline.joblib"))


@click.command()
@click.argument(
    "input_data_dir", default="ml/data/interim", type=click.Path(exists=True)
)
@click.argument("output_model_dir", default="ml/models", type=click.Path())
def main(input_data_dir: str, output_model_dir: str) -> None:
    """
    Run data processing scripts to turn cleaned data from (../interim) into
    training data ready to be trained (saved in ../processed).
    """
    logger.info(f"Read from {input_data_dir}, write to {output_model_dir}.")
    pipeline(input_data_dir, output_model_dir)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-parameter
    main()
