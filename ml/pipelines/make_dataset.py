"""
Data Preprocessing Pipeline

This script defines a data preprocessing pipeline to convert raw data from one
format to another. It reads data from a CSV file in the 'input_dir', processes
it, and saves the
processed data to a new CSV file in the 'output_dir'.

Usage:
    To run this script, use the following command:
    ```
    python preprocess_data.py [INPUT_DIR] [OUTPUT_DIR]
    ```

Arguments:
    INPUT_DIR (str): The directory containing the raw input data
    (default: "ml/data/raw").
    OUTPUT_DIR (str): The directory where the processed data will be saved
    (default: "ml/data/interim").

Example:
    ```
    python preprocess_data.py ml/data/raw ml/data/interim
    ```
"""
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger


def pipeline(input_dir: str, output_dir: str) -> None:
    """
    Preprocess raw data and save it to a new CSV file.

    :param str input_dir: The directory containing the raw input data
    :param str output_dir: The directory where the processed data will be saved
    """
    logger.info("Start preprocessing data.")
    data_in_path = Path(input_dir, "train.csv")
    data_out_path = Path(output_dir, "train_data.csv")

    logger.info(f"Reading data from {data_in_path}")
    data = pd.read_csv(data_in_path)
    logger.info(f"Saving data to {data_out_path}")
    data.to_csv(data_out_path, index=False)
    logger.info(f"Processed data saved to {output_dir}")


@click.command()
@click.argument("input_dir", default="ml/data/raw", type=click.Path(exists=True))
@click.argument("output_dir", default="ml/data/interim", type=click.Path())
def main(input_dir: str, output_dir: str) -> None:
    """
    Run data processing scripts to preprocess raw data from (../raw) into
    cleaned data ready for analysis (saved in ../interim).
    """
    logger.info(f"Read from {input_dir}, write to {output_dir}.")
    pipeline(input_dir, output_dir)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-parameter
    main()
