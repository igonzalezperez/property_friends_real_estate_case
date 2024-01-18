import click
from pathlib import Path

from loguru import logger
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from joblib import dump


def pipeline(input_data_dir, output_model_dir):
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
def main(input_data_dir, output_model_dir):
    """Runs data processing scripts to turn cleaned data from (../interim) into
    training data ready to be trained (saved in ../processed).
    """
    logger.info(f"Read from {input_data_dir}, write to {output_model_dir}.")
    pipeline(input_data_dir, output_model_dir)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-paramete
    main()
