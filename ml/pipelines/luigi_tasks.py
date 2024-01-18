"""
Data Processing Pipeline with Luigi

This module defines a data processing pipeline using Luigi, a Python library
for building complex data pipelines.
The pipeline consists of three tasks: MakeDataset, BuildFeatures, and
TrainModel, which are executed in sequence.

Tasks:
- MakeDataset: Reads raw data and saves it as a processed dataset.
- BuildFeatures: Performs feature engineering on the dataset and saves a
preprocessing pipeline.
- TrainModel: Trains a machine learning model using the preprocessed dataset
and saves the trained model.

Usage:
To run the pipeline, execute this module. For example:
    (poetry run) python module_name.py

Note: You can customize the input/output directories and other parameters in
this script as needed.
"""
from pathlib import Path

import luigi
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

RAW_DATA_FOLDER = "ml/data/raw"
INTERIM_DATA_FOLDER = "ml/data/interim"
MODELS_FOLDER = "ml/models"

INPUT_DATA_FILE = "train.csv"
PREPROC_PIPELINE_FILE = "preproc_pipeline.joblib"
MODEL_FILE = "model.joblib"


class MakeDataset(luigi.Task):  # type: ignore[misc]
    """
    Luigi task for creating the dataset from raw files.
    """

    input_dir = luigi.Parameter(default=RAW_DATA_FOLDER)
    output_dir = luigi.Parameter(default=INTERIM_DATA_FOLDER)

    def output(self) -> luigi.LocalTarget:
        """
        Define task output.

        :return luigi.LocalTarget: Target where output data will be saved
        """
        return luigi.LocalTarget(Path(self.output_dir, INPUT_DATA_FILE))

    def run(self) -> None:
        """
        Run task logic. Read data from input, clean it and prepareit, then
        save it.
        """
        data_in_path = Path(self.input_dir, INPUT_DATA_FILE)
        data_out_path = self.output().path

        logger.info(f"Reading data from {data_in_path}")
        data = pd.read_csv(data_in_path)

        logger.info(f"Saving data to {data_out_path}")
        data.to_csv(data_out_path, index=False)
        logger.success("MakeDataset process ran successfully")


class BuildFeatures(luigi.Task):  # type: ignore[misc]
    """
    Luigi task for creating data features.
    """

    input_data_dir = luigi.Parameter(default=INTERIM_DATA_FOLDER)
    output_model_dir = luigi.Parameter(default=MODELS_FOLDER)

    def requires(self) -> MakeDataset:
        """
        Required task to run before this one

        :return MakeDataset: MakeDataset task
        """
        return MakeDataset()

    def output(self) -> luigi.LocalTarget:
        """
        Define task output.

        :return luigi.LocalTarget: Target where output data will be saved
        """
        return luigi.LocalTarget(
            Path(
                self.output_model_dir,
                PREPROC_PIPELINE_FILE,
            )
        )

    def run(self) -> None:
        """
        Run task logic. Preprocess model features then save data and artifacts
        """
        data_path = Path(self.input_data_dir, INPUT_DATA_FILE)

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

        steps = [("preprocessor", preprocessor)]
        preproc_pipeline = Pipeline(steps).fit(data[feature_cols], data[target])
        model_path = self.output().path

        logger.info(f"Saving preprocessing pipeline to: {model_path}")
        dump(preproc_pipeline, model_path)
        logger.success("BuildFeatures process ran successfully")


class TrainModel(luigi.Task):  # type: ignore[misc]
    """
    Luigi task for training predictive model.
    """

    input_data_dir = luigi.Parameter(default=INTERIM_DATA_FOLDER)
    input_model_dir = luigi.Parameter(default=MODELS_FOLDER)
    output_model_dir = luigi.Parameter(default=MODELS_FOLDER)

    def requires(self) -> MakeDataset:
        """
        Required task to run before this one

        :return MakeDataset: MakeDataset task
        """
        return MakeDataset()

    def output(self) -> luigi.LocalTarget:
        """
        Define task output.

        :return luigi.LocalTarget: Target where output data will be saved
        """
        return luigi.LocalTarget(Path(self.output_model_dir, MODEL_FILE))

    def run(self) -> None:
        """
        Run task logic. Read data from input, train model, then
        save it.
        """
        data_path = Path(self.input_data_dir, INPUT_DATA_FILE)
        preproc_path = Path(self.input_model_dir, PREPROC_PIPELINE_FILE)

        logger.info(f"Reading data from: {data_path}")
        logger.info(f"Reading preprocessing pipeline from: {preproc_path}")

        data = pd.read_csv(data_path)

        categorical_cols = ["type", "sector"]
        target = "price"
        feature_cols = [col for col in data.columns if col != target]

        categorical_transformer = TargetEncoder()

        preprocessor = ColumnTransformer(
            transformers=[("categorical", categorical_transformer, categorical_cols)]
        )

        steps = [
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    **{
                        "learning_rate": 0.01,
                        "n_estimators": 300,
                        "max_depth": 5,
                        "loss": "absolute_error",
                    }
                ),
            ),
        ]
        pipeline = Pipeline(steps)
        logger.info(f"Training pipeline: \n{pipeline}")

        pipeline.fit(data[feature_cols], data[target])
        model_path = self.output().path

        logger.info(f"Saving model to {model_path}")
        dump(pipeline, model_path)
        logger.success("TrainModel process ran successfully")


if __name__ == "__main__":
    # Run all tasks, considering TrainModel requires previous tasks
    luigi.run(["TrainModel", "--local-scheduler"])
