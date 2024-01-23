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
from tabulate import tabulate
import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from sklearn.model_selection import train_test_split
from sqlalchemy import (
    create_engine,
    Table,
    Column,
    Boolean,
    Text,
    Numeric,
    MetaData,
    Integer,
)
from ml.pipelines.utils import DB_CONN_STR, get_pipeline_config, get_table_as_df
from sqlalchemy.orm import sessionmaker
from starlette.config import Config

load_dotenv(find_dotenv())

config = Config(".env")
SHUFFLE_RAW_DATA: bool = config("SHUFFLE_RAW_DATA", cast=bool, default=False)
USE_DB: bool = config("USE_DB", cast=bool, default=True)


def raw_data_table() -> (Table, MetaData):
    metadata = MetaData()
    table = Table(
        "raw_data",
        metadata,
        Column("type", Text),
        Column("sector", Text),
        Column("net_usable_area", Numeric),
        Column("net_area", Numeric),
        Column("n_rooms", Numeric),
        Column("n_bathroom", Numeric),
        Column("latitude", Numeric),
        Column("longitude", Numeric),
        Column("price", Numeric),
        Column("is_train", Boolean),  # Preserve original train-test split
    )
    return table, metadata


def init_raw_data_table(db_conn_str: str, input_data_df: pd.DataFrame) -> None:
    engine = create_engine(db_conn_str)
    table, metadata = raw_data_table()
    logger.info("Creating `raw_data` table if it doesn't exist")
    metadata.create_all(engine)
    _session = sessionmaker(engine)
    session = _session()
    query = session.query(table).limit(5)
    df = pd.read_sql(query.statement, engine)
    if df.empty:
        logger.info("Table `raw_data` is empty, inserting data")
        # Insert data if table is empty
        input_data_df.to_sql(
            name="raw_data",
            con=engine,
            if_exists="append",
            index=False,
        )
        df = pd.read_sql(query.statement, engine)
    else:
        logger.info("Table `raw_data` already has data")
    logger.info(
        f"""\n
        {tabulate(
            df.head().to_numpy().tolist(),
            headers=df.columns.tolist(),
            tablefmt="grid",
        )}
                """
    )


def pipeline(
    input_dir: str,
    input_file: str,
    input_test_file: str,
    output_dir: str,
) -> None:
    """
    Preprocess raw data and save it to a new CSV file.

    :param str input_dir: Directory of input
    :param str input_file: File name of input
    :param str input_test_file: File name of test input
    :param str output_path: Directory of output
    """
    input_path = Path(input_dir, input_file)
    input_test_path = Path(input_dir, input_test_file)
    output_path = str(
        Path(output_dir, f"{os.getenv('INTERIM_DATA_PREFIX')}_{input_file}")
    )
    output_test_path = str(
        Path(output_dir, f"{os.getenv('INTERIM_DATA_PREFIX')}_{input_test_file}")
    )
    # Load data from db
    if USE_DB:
        try:
            # Write local data to db table if it's empty
            data = pd.read_csv(input_path)
            data_test = pd.read_csv(input_test_path)
            data["is_train"] = True
            data_test["is_train"] = False
            df = pd.concat([data, data_test], axis=0)
            init_raw_data_table(DB_CONN_STR, df)

            # Read raw_data table from db
            logger.info("Reading data from DB")
            df = get_table_as_df(DB_CONN_STR, "raw_data")

            # Keep original train-test split
            data = df[df["is_train"]].drop("is_train", axis=1)
            data_test = df[~df["is_train"]].drop("is_train", axis=1)
        except FileNotFoundError:
            logger.warning("No local data to populate db table `raw_data`")

    # Load data from files
    else:
        if not Path(input_dir, input_file).exists():
            err_msg = f"File '{Path(input_dir, input_file)}' doesn't exist"
            raise FileNotFoundError(err_msg)
        if not Path(input_dir, input_test_file).exists():
            err_msg = f"File '{Path(input_dir, input_test_file)}' doesn't exist"
            raise FileNotFoundError(err_msg)

        # Only read if haven't been read earlier, when setting up the db
        if "data" not in locals():
            logger.info(f"Reading data from {input_path}")
            data = pd.read_csv(input_path)
            if "is_train" in data.columns:
                data = data.drop("is_train", axis=1)
        if "data_test" not in locals():
            logger.info(f"Reading data from {input_test_path}")
            data_test = pd.read_csv(input_test_path)
            if "is_train" in data_test.columns:
                data_test = data_test.drop("is_train", axis=1)
    if SHUFFLE_RAW_DATA:
        logger.info(
            """Merging train-test data and shuffling rows to create new split"""
        )
        data = pd.concat([data, data_test], ignore_index=True)
        data, data_test = train_test_split(data, test_size=0.3, random_state=1)
    else:
        logger.info("Keeping original train-test split")
    params = get_pipeline_config()
    target_col = params["target_col"]

    # Drop cols with 0 or less in the target
    thresh = os.getenv("MIN_TARGET_THRESHOLD", default="0")
    thresh = int(thresh)
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
    logger.success("Successfully ran make_dataset")


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
