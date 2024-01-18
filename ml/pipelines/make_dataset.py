# -*- coding: utf-8 -*-
import click
from pathlib import Path
import pandas as pd
from loguru import logger
from dotenv import find_dotenv, load_dotenv


def pipeline(input_dir, output_dir):
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
def main(input_dir, output_dir):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info(f"Read from {input_dir}, write to {output_dir}.")
    pipeline(input_dir, output_dir)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-paramete
    main()
