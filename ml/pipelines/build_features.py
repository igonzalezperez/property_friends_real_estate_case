"""
Data Processing Pipeline

This script defines a data processing pipeline defined in
ml/pipelines/pipeline_config.yml to transform input data for training and
testing.

Usage:
    To run this script, use the following command:
    ```
    (poetry run) python ml/pipelines/build_features.py [INPUT_DIR]
    [INPUT_FILE] [INPUT_TEST_FILE] [OUTPUT_DATA_DIR] [OUTPUT_MODEL_DIR]
    ```
"""
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import dump
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml.pipelines.utils import get_pipeline_config

load_dotenv(find_dotenv())


def pipeline(
    input_dir: str,
    input_file: str,
    input_test_file: str,
    output_data_dir: str,
    output_model_dir: str,
) -> None:
    """
    Build a data preprocessing pipeline to transform input data.

    :param str input_dir: Input data directory
    :param str input_file: Input file name
    :param str input_test_file: Input test file name
    :param str output_data_path: Output data file path
    :param str output_model_path: Output model file path
    """
    output_data_path = str(
        Path(
            output_data_dir,
            f"{str(os.getenv('PROCESSED_DATA_PREFIX'))}_{input_file.split('_')[-1]}",
        ),
    )
    output_test_data_path = str(
        Path(
            output_data_dir,
            f"{str(os.getenv('PROCESSED_DATA_PREFIX'))}_{input_test_file.split('_')[-1]}",
        ),
    )
    output_model_path = str(
        Path(
            output_model_dir,
            str(os.getenv("PREPROC_FILE_NAME")),
        )
    )
    is_test = True
    if not Path(input_dir, input_file).exists():
        err_msg = f"File '{Path(input_dir, input_file)}' doesn't exist"
        raise FileNotFoundError(err_msg)
    if not Path(input_dir, input_file).exists():
        err_msg = f"File '{Path(input_dir, input_test_file)}' doesn't exist"
        logger.warning(err_msg)
        is_test = False
    logger.info("Load data pipeline config.")
    params = get_pipeline_config()
    logger.info("Start processing data.")
    input_path = Path(input_dir, input_file)
    input_test_path = Path(input_dir, input_test_file)

    logger.info(f"Reading data from: {input_path}")
    data = pd.read_csv(input_path)
    transformers = []
    feature_cols = []

    # Add col transformers if defined
    if "categorical_transform" in params:
        tr_cat = (
            "categorical",
            params["categorical_transform"](),
            params["cat_cols"],
        )
        transformers.append(tr_cat)
        feature_cols.extend(params["cat_cols"])
    if "numerical_transform" in params:
        tr_num = (
            "numerical",
            params["numerical_transform"](),
            params["num_cols"],
        )
        transformers.append(tr_num)
        feature_cols.extend(params["num_cols"])

    logger.info(f"Data cols: {data.columns.tolist()}")
    logger.info(f"Feature cols: {feature_cols}")
    logger.info(f"Categorical cols: {params['cat_cols']}")
    logger.info(f"Numerical cols: {params['num_cols']}")
    logger.info(f"Target col: {params['target_col']}")

    preprocessor = ColumnTransformer(transformers=transformers)

    steps = [("preprocessor", preprocessor)]
    logger.info("Fitting preprocessing pipeline")
    preproc_pipeline = Pipeline(steps).fit(
        data[feature_cols],
        data[params["target_col"]],
    )
    logger.info(f"Succesfully fitted pipeline: \n{preproc_pipeline}")

    preproc_data = preproc_pipeline.transform(data[feature_cols])
    preproc_data = pd.DataFrame(preproc_data, columns=feature_cols)
    preproc_data = pd.concat([preproc_data, data[params["target_col"]]], axis=1)

    logger.info(f"Saving preprocessed data to: {output_data_path}")
    preproc_data.to_csv(output_data_path, index=False)
    if is_test:
        logger.info(f"Reading data from: {input_test_path}")
        data_test = pd.read_csv(input_test_path)
        preproc_test_data = preproc_pipeline.transform(data_test[feature_cols])
        preproc_test_data = pd.DataFrame(preproc_test_data, columns=feature_cols)
        preproc_test_data = pd.concat(
            [preproc_test_data, data[params["target_col"]]], axis=1
        )
        logger.info(f"Saving preprocessed data to: {output_test_data_path}")
        preproc_test_data.to_csv(output_test_data_path, index=False)

    logger.info(f"Saving preprocessing pipeline to: {output_model_path}")
    dump(preproc_pipeline, output_model_path)

    logger.success("Successfully ran build_features")


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
    "--output-data-dir",
    "output_data_dir",
    default=Path(str(os.getenv("ML_DATA_DIR")), "processed"),
    type=click.Path(exists=True),
)
@click.option(
    "--output-model-dir",
    "output_model_dir",
    default=os.getenv("ML_MODELS_DIR"),
    type=click.Path(exists=True),
)
def run(
    input_dir: str,
    input_file: str,
    input_test_file: str,
    output_data_dir: str,
    output_model_dir: str,
) -> None:
    """
    Run data processing scripts to turn input data from (../interim) into
    processed data ready to be trained (saved in ../processed). As well as
    trained pipeline artifact (saved in ../models)
    """

    pipeline(
        input_dir,
        input_file,
        input_test_file,
        output_data_dir,
        output_model_dir,
    )


if __name__ == "__main__":
    # pylint: disable = no-value-for-parameter
    run()
