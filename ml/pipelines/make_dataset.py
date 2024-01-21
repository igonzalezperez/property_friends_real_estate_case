"""
Data Loading Pipeline

This script defines a data loading pipeline to convert raw data from one
format to and intermediate one. It reads data from a CSV file in the
'input_dir' with name 'input_file', processes it, and saves the processed data
to a new CSV file in the 'output_dir'.

Usage:
    To run this script, use the following command:
    ```
    (poetry run) python ml/pipelines/make_dataset.py [INPUT_DIR]
    [INPUT_FILE] [INPUT_TEST_FILE] [OUTPUT_DIR]
    ```
"""
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from sklearn.model_selection import train_test_split

from ml.pipelines.utils import get_pipeline_config

load_dotenv(find_dotenv())


def pipeline(
    input_dir: str,
    input_file: str,
    input_test_file: str,
    output_dir: str,
) -> None:
    """
    Preprocess raw data and save it to a new CSV file.

    :param Union[str, Path] input_dir: Directory of input
    :param Union[str, Path] input_file: File name of input
    :param Union[str, Path] output_path: Directory of output
    """
    if not Path(input_dir, input_file).exists():
        err_msg = f"File '{Path(input_dir, input_file)}' doesn't exist"
        raise FileNotFoundError(err_msg)
    if not Path(input_dir, input_test_file).exists():
        err_msg = f"File '{Path(input_dir, input_test_file)}' doesn't exist"
        raise FileNotFoundError(err_msg)
    output_path = str(
        Path(output_dir, f"{os.getenv('INTERIM_DATA_PREFIX')}_{input_file}")
    )
    output_test_path = str(
        Path(output_dir, f"{os.getenv('INTERIM_DATA_PREFIX')}_{input_test_file}")
    )
    logger.info(f"Read from {input_file}, write to {output_path}.")
    logger.info("Start preprocessing data.")
    input_path = Path(input_dir, input_file)
    input_test_path = Path(input_dir, input_test_file)

    logger.info(f"Reading data from {input_path}")
    logger.info(f"Reading data from {input_test_path}")
    data = pd.read_csv(input_path)
    data_test = pd.read_csv(input_test_path)

    if os.getenv("SHUFFLE_RAW_DATA"):
        logger.info("Merging train and test and shuffling rows to create new split")
        data = pd.concat([data, data_test], ignore_index=True)
        data, data_test = train_test_split(data, test_size=0.3, random_state=1)
    else:
        logger.info("Keeping original train-test split")
    params = get_pipeline_config()
    target_col = params["target_col"]

    # Drop cols with 0 or less in the target
    thresh = os.getenv("MIN_TARGET_THRESHOLD", default=0)
    logger.info(
        f"Dropping all rows with target less or equal than {thresh}",
    )
    drop_data = data[data[target_col] <= thresh]
    drop_test_data = data_test[data_test[target_col] <= thresh]

    logger.info(f"Dropped {len(drop_data)} train cols")
    logger.info(f"Dropped {len(drop_test_data)} test cols")

    data = data[data[target_col] > thresh]
    data_test = data_test[data_test[target_col] > thresh]

    logger.info(f"Saving data to {output_path}")
    logger.info(f"Saving data to {output_test_path}")
    data.to_csv(output_path, index=False)
    data_test.to_csv(output_test_path, index=False)
    logger.success("Successfully ran MakeDataset")


@click.command()
@click.option(
    "--input-dir",
    "input_dir",
    default=Path(str(os.getenv("ML_DATA_DIR")), "raw"),
    type=click.Path(exists=True),
)
@click.option(
    "--input-file",
    "input_file",
    default=os.getenv("TRAIN_FILE_NAME"),
)
@click.option(
    "--input-test-file",
    "input_test_file",
    default=os.getenv("TEST_FILE_NAME"),
)
@click.option(
    "--output-dir",
    "output_dir",
    default=Path(str(os.getenv("ML_DATA_DIR")), "interim"),
    type=click.Path(exists=True),
)
def run(
    input_dir: str,
    input_file: str,
    input_test_file: str,
    output_dir: str,
) -> None:
    """
    Run data processing scripts to preprocess raw data from (../raw) into
    cleaned data ready for analysis (saved in ../interim).
    """
    pipeline(input_dir, input_file, input_test_file, output_dir)


if __name__ == "__main__":
    # pylint: disable = no-value-for-parameter
    run()
