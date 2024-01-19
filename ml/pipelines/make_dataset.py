"""
Data Preprocessing Pipeline

This script defines a data preprocessing pipeline to convert raw data from one
format to and intermediate one. It reads data from a CSV file in the
'input_dir' with name 'input_file', processes it, and saves the processed data
to a new CSV file in the 'output_dir'. All the variables can be passed as
arguments, but they have defaults too.

Usage:

    ```
    python preprocess_data.py --input-dir . --input-file . -output-dir .
    ```

Arguments:
    INPUT_DIR (str): The directory containing the raw input data
    (default: "ml/data/raw").
    INPUT_FILE (str): The file name containing the raw input data
    (default: "train.csv").
    OUTPUT_DIR (str): The directory where the processed data will be saved
    (default: "ml/data/interim").

Example:
    To run this script, use the following command:
    ```
    poetry run python ml/pipelines/make_dataset.py --input-dir ml/data/raw
    --input-file train.csv --output-dir ml/data/interim
    ```
"""
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from app.core.config import ML_DATA_DIR


def pipeline(
    input_dir: str,
    input_file: str,
    output_path: str,
) -> None:
    """
    Preprocess raw data and save it to a new CSV file.

    :param Union[str, Path] input_dir: Directory of input
    :param Union[str, Path] input_file: File name of input
    :param Union[str, Path] output_path: Directory of output
    """
    logger.info("Start preprocessing data.")
    input_path = Path(input_dir, input_file)
    logger.info(f"Reading data from {input_path}")
    data = pd.read_csv(input_path)
    logger.info(f"Saving data to {output_path}")
    data.to_csv(output_path, index=False)
    logger.success("Successfully ran MakeDataset")


@click.command()
@click.option(
    "--input-dir",
    "input_dir",
    default=Path(ML_DATA_DIR, "raw"),
    type=click.Path(exists=True),
)
@click.option(
    "--input-file",
    "input_file",
    default="train.csv",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=Path(ML_DATA_DIR, "interim"),
    type=click.Path(exists=True),
)
def run(
    input_dir: str,
    input_file: str,
    output_dir: str,
) -> None:
    """
    Run data processing scripts to preprocess raw data from (../raw) into
    cleaned data ready for analysis (saved in ../interim).
    """
    logger.info("Running make dataset pipeline")
    if not Path(input_dir, input_file).exists():
        err_msg = f"File '{Path(input_dir, input_file)}' doesn't exist"
        raise FileNotFoundError(err_msg)
    output_path = str(Path(output_dir, f"preprocessed_{input_file}"))
    logger.info(f"Read from {input_file}, write to {output_path}.")
    pipeline(input_dir, input_file, output_path)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-parameter
    run()
