# -*- coding: utf-8 -*-
import click
from pathlib import Path

from loguru import logger
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from joblib import dump, load


def pipeline(input_data_dir, input_model_dir, output_model_dir):
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
def main(input_data_dir, input_model_dir, output_model_dir):
    """Runs data processing scripts to turn cleaned data from (../interim) into
    training data ready to be trained (saved in ../processed).
    """
    logger.info(
        f"Read from {input_data_dir}, {input_model_dir}, write to {output_model_dir}."
    )
    pipeline(input_data_dir, input_model_dir, output_model_dir)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-paramete
    main()
