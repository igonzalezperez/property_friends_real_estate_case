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
import mlflow
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FOLDER = "ml/data"
RAW_DATA_FOLDER = "ml/data/raw"
INTERIM_DATA_FOLDER = "ml/data/interim"
MODELS_FOLDER = "ml/models"

INPUT_DATA_FILE = "train.csv"
PREPROC_DATA_FILE = "processed_train.csv"
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
    output_data_dir = luigi.Parameter(default=Path(DATA_FOLDER, "processed"))
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
                self.output_data_dir,
                PREPROC_DATA_FILE,
            )
        )

    def run(self) -> None:
        """
        Run task logic. Preprocess model features then save data and artifacts
        """
        data_in_path = Path(self.input_data_dir, INPUT_DATA_FILE)
        data_out_path = Path(self.output_data_dir, PREPROC_DATA_FILE)
        pipeline_out_path = Path(self.output_data_dir, PREPROC_PIPELINE_FILE)

        logger.info(f"Reading data from: {data_in_path}")
        data = pd.read_csv(data_in_path)

        categorical_cols = ["type", "sector"]
        target = "price"
        numerical_cols = [
            i for i in data.columns if i not in [*categorical_cols, target]
        ]
        feature_cols = [*categorical_cols, *numerical_cols]

        logger.info(f"Data cols: {data.columns.tolist()}")
        logger.info(f"Feature cols: {feature_cols}")
        logger.info(f"Categorical cols: {categorical_cols}")
        logger.info(f"Numerical cols: {numerical_cols}")
        logger.info(f"Target col: {target}")

        categorical_transformer = TargetEncoder()
        numerical_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, categorical_cols),
                ("numerical", numerical_transformer, numerical_cols),
            ]
        )

        steps = [("preprocessor", preprocessor)]
        logger.info("Fitting preprocessing pipeline")
        preproc_pipeline = Pipeline(steps).fit(
            data[feature_cols],
            data[target],
        )
        preproc_data = preproc_pipeline.transform(data[feature_cols])
        preproc_data = pd.DataFrame(preproc_data, columns=feature_cols)
        preproc_data = pd.concat([preproc_data, data[target]], axis=1)

        logger.info(f"Succesfully fitted pipeline: \n{preproc_pipeline}")

        logger.info(f"Saving preprocessing pipeline to: {pipeline_out_path}")

        dump(preproc_pipeline, pipeline_out_path)

        logger.info(f"Saving preprocessed data to: {data_out_path}")
        preproc_data.to_csv(data_out_path, index=False)
        logger.success("BuildFeatures process ran successfully")


class TrainModel(luigi.Task):  # type: ignore[misc]
    """
    Luigi task for training predictive model.
    """

    input_data_dir = luigi.Parameter(default=Path(DATA_FOLDER, "processed"))
    output_model_dir = luigi.Parameter(default=MODELS_FOLDER)

    def requires(self) -> BuildFeatures:
        """
        Required task to run before this one

        :return BuildFeatures: BuildFeatures task
        """
        return BuildFeatures()

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
        data_path = Path(self.input_data_dir, PREPROC_DATA_FILE)

        logger.info(f"Reading processed data from: {data_path}")

        data = pd.read_csv(data_path)

        target = "price"
        feature_cols = [i for i in data.columns if i != target]

        logger.info(f"Feature cols: {feature_cols}")
        logger.info(f"Target col: {target}")

        with mlflow.start_run():
            params = {
                "learning_rate": 0.01,
                "n_estimators": 300,
                "max_depth": 5,
                "loss": "absolute_error",
            }
            for k, v in params.items():
                mlflow.log_param(k, v)
            model = GradientBoostingRegressor(**params)

            logger.info("Training model")
            model.fit(data[feature_cols], data[target])
            logger.info(f"Succesfully trained model: \n{model}")
            model_path = self.output().path

            logger.info(f"Saving model to {model_path}")
            dump(model, model_path)
            mlflow.log_artifact(model_path)
            logger.success("TrainModel process ran successfully")


if __name__ == "__main__":
    # Run all tasks, considering TrainModel requires previous tasks
    luigi.run(["TrainModel", "--local-scheduler"])
