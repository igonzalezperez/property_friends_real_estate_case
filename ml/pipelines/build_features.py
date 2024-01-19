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
import yaml
from dotenv import find_dotenv, load_dotenv
from joblib import dump
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from app.core.config import ML_DATA_DIR, ML_DIR, ML_MODELS_DIR

with open(
    Path(ML_DIR, "pipelines", "pipeline_config.yml"), "r", encoding="utf-8"
) as file:
    config = yaml.safe_load(file)

NUM_COLS = config["data_catalog"]["columns"]["features"]["numerical"]
CAT_COLS = config["data_catalog"]["columns"]["features"]["categorical"]
FEATURE_COLS = [*CAT_COLS, *NUM_COLS]
TARGET_COL = config["data_catalog"]["columns"]["target"]

CAT_TRANSFORM_PATH = config["model_pipeline"]["preprocessor"]["transformers"][
    "categorical"
]
NUM_TRANSFORM_PATH = config["model_pipeline"]["preprocessor"]["transformers"][
    "numerical"
]

CAT_TRANSFORM_MODULE, CAT_TRANSFORM_CLASS = CAT_TRANSFORM_PATH.rsplit(".", 1)
NUM_TRANSFORM_MODULE, NUM_TRANSFORM_CLASS = NUM_TRANSFORM_PATH.rsplit(".", 1)

CAT_TRANSFORM_ = __import__(CAT_TRANSFORM_MODULE, fromlist=[CAT_TRANSFORM_CLASS])
NUM_TRANSFORM_ = __import__(NUM_TRANSFORM_MODULE, fromlist=[NUM_TRANSFORM_CLASS])

CAT_TRANSFORM = getattr(CAT_TRANSFORM_, CAT_TRANSFORM_CLASS)()
NUM_TRANSFORM = getattr(NUM_TRANSFORM_, NUM_TRANSFORM_CLASS)()


def pipeline(
    input_dir: str,
    input_file: str,
    output_data_path: str,
    output_model_path: str,
) -> None:
    """
    Build a data preprocessing pipeline to transform input data.

    :param Union[str, Path] input_dir: Input data directory
    :param Union[str, Path] input_file: Input file name
    :param Union[str, Path] output_data_path: Output data file path
    :param Union[str, Path] output_model_path: Output model file path
    """
    logger.info("Start processing data.")
    input_path = Path(input_dir, input_file)
    logger.info(f"Reading data from: {input_path}")
    data = pd.read_csv(input_path)

    logger.info(f"Data cols: {data.columns.tolist()}")
    logger.info(f"Feature cols: {FEATURE_COLS}")
    logger.info(f"Categorical cols: {CAT_COLS}")
    logger.info(f"Numerical cols: {NUM_COLS}")
    logger.info(f"Target col: {TARGET_COL}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                CAT_TRANSFORM,
                CAT_COLS,
            ),
            (
                "numerical",
                NUM_TRANSFORM,
                NUM_COLS,
            ),
        ]
    )

    steps = [("preprocessor", preprocessor)]
    logger.info("Fitting preprocessing pipeline")
    preproc_pipeline = Pipeline(steps).fit(
        data[FEATURE_COLS],
        data[TARGET_COL],
    )
    preproc_data = preproc_pipeline.transform(data[FEATURE_COLS])
    preproc_data = pd.DataFrame(preproc_data, columns=FEATURE_COLS)
    preproc_data = pd.concat([preproc_data, data[TARGET_COL]], axis=1)

    logger.info(f"Succesfully fitted pipeline: \n{preproc_pipeline}")

    logger.info(f"Saving preprocessing pipeline to: {output_model_path}")

    dump(preproc_pipeline, output_model_path)

    logger.info(f"Saving preprocessed data to: {output_data_path}")
    preproc_data.to_csv(output_data_path, index=False)
    logger.success("Successfully ran BuildFeatures")


@click.command()
@click.option(
    "--input-dir",
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
    "--output-data-dir",
    "output_data_dir",
    default=Path(ML_DATA_DIR, "processed"),
    type=click.Path(exists=True),
)
@click.option(
    "--output-model-dir",
    "output_model_dir",
    default=ML_MODELS_DIR,
    type=click.Path(exists=True),
)
def main(
    input_dir: str,
    input_file: str,
    output_data_dir: str,
    output_model_dir: str,
) -> None:
    """
    Run data processing scripts to turn input data from (../interim) into
    processed data ready to be trained (saved in ../processed). As well as
    trained pipeline artifact (saved in ../models)
    """
    output_data_path = str(
        Path(output_data_dir, f"processed_{input_file.split('_')[-1]}"),
    )
    output_model_path = str(Path(output_model_dir, "preproc_pipeline.joblib"))
    if not Path(input_dir, input_file).exists():
        err_msg = f"File '{Path(input_dir, input_file)}' doesn't exist"
        raise FileNotFoundError(err_msg)
    pipeline(input_dir, input_file, output_data_path, output_model_path)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-parameter
    main()
