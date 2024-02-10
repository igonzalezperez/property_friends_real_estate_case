"""
ML Pipeline with Luigi

This module defines an ML pipeline using Luigi.
The pipeline consists of three tasks: MakeDataset, BuildFeatures, and
TrainModel, which are executed in sequence.

Tasks:
- MakeDataset: Reads raw data and saves it as a preprocessed dataset.
- BuildFeatures: Performs feature engineering on the dataset and saves a
fitted pipeline as well as processed data.
- TrainModel: Trains a machine learning model using the previously
fitted pipeline and saves the trained model.

Usage:
To run the pipeline, execute this module run:
    (poetry run) python ml/pipelines/luigi_tasks.py
"""

import os
from pathlib import Path

import luigi
from dotenv import find_dotenv, load_dotenv

from ml.pipelines import build_features, make_dataset, train_model

load_dotenv(find_dotenv())


class MakeDataset(luigi.Task):  # type: ignore[misc]
    """
    Luigi task for creating the dataset from raw files.
    """

    input_dir = luigi.Parameter(
        default=str(
            Path(
                str(os.getenv("ML_DATA_DIR")),
                "raw",
            )
        )
    )
    input_file = luigi.Parameter(default=os.getenv("TRAIN_FILE_NAME"))
    output_dir = luigi.Parameter(
        default=str(
            Path(
                str(os.getenv("ML_DATA_DIR")),
                "interim",
            )
        )
    )
    input_test_file = luigi.Parameter(default=os.getenv("TEST_FILE_NAME"))
    output_dir = luigi.Parameter(
        default=str(
            Path(
                str(os.getenv("ML_DATA_DIR")),
                "interim",
            )
        )
    )

    def output(self) -> luigi.LocalTarget:
        """
        Define task output.

        :return luigi.LocalTarget: Target where output data will be saved
        """
        output_path = str(
            Path(
                self.output_dir,
                f"{str(os.getenv('INTERIM_DATA_PREFIX'))}_{self.input_file}",
            )
        )
        output_test_path = str(
            Path(
                self.output_dir,
                f"{str(os.getenv('INTERIM_DATA_PREFIX'))}_{self.input_test_file}",
            )
        )
        return (luigi.LocalTarget(output_path), luigi.LocalTarget(output_test_path))

    def run(self) -> None:
        """
        Run task logic. Read data from input, clean it and prepareit, then
        save it.
        """
        make_dataset.pipeline(
            input_dir=str(self.input_dir),
            input_file=str(self.input_file),
            input_test_file=str(self.input_test_file),
            output_dir=str(self.output_dir),
        )


class BuildFeatures(luigi.Task):  # type: ignore[misc]
    """
    Luigi task for creating data features.
    """

    input_dir = luigi.Parameter(Path(str(os.getenv("ML_DATA_DIR")), "interim"))
    input_file = luigi.Parameter(
        default=f"{str(os.getenv('INTERIM_DATA_PREFIX'))}_{str(os.getenv('TRAIN_FILE_NAME'))}",
    )
    input_test_file = luigi.Parameter(
        default=f"{str(os.getenv('INTERIM_DATA_PREFIX'))}_{str(os.getenv('TEST_FILE_NAME'))}",
    )
    output_data_dir = luigi.Parameter(
        default=Path(str(os.getenv("ML_DATA_DIR")), "processed")
    )
    output_model_dir = luigi.Parameter(default=str(os.getenv("ML_MODELS_DIR")))

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
        prefix = f"{str(os.getenv('PROCESSED_DATA_PREFIX'))}"
        name = f"{str(self.input_file).split('_', maxsplit=1)[-1]}"
        test_name = f"{str(self.input_test_file).split('_', maxsplit=1)[-1]}"
        output_data_path = str(
            Path(self.output_data_dir, f"{prefix}_{name}"),
        )
        output_test_data_path = str(
            Path(
                self.output_data_dir,
                f"{prefix}_{test_name}",
            ),
        )
        output_model_path = str(
            Path(
                self.output_model_dir,
                str(os.getenv("PREPROC_FILE_NAME")),
            )
        )
        return (
            luigi.LocalTarget(output_data_path),
            luigi.LocalTarget(output_test_data_path),
            luigi.LocalTarget(output_model_path),
        )

    def run(self) -> None:
        """
        Run task logic. Preprocess model features then save data and artifacts
        """
        build_features.pipeline(
            input_dir=self.input_dir,
            input_file=self.input_file,
            input_test_file=self.input_test_file,
            output_data_dir=self.output_data_dir,
            output_model_dir=self.output_model_dir,
        )


class TrainModel(luigi.Task):  # type: ignore[misc]
    """
    Luigi task for training predictive model.
    """

    input_dir = Path(str(os.getenv("ML_DATA_DIR")), "interim")
    input_file = f"{os.getenv('INTERIM_DATA_PREFIX')}_{os.getenv('TRAIN_FILE_NAME')}"
    input_test_file = (
        f"{os.getenv('INTERIM_DATA_PREFIX')}_{os.getenv('TEST_FILE_NAME')}"
    )
    input_pipeline_dir = os.getenv("ML_MODELS_DIR")
    input_pipeline_file = os.getenv("PREPROC_FILE_NAME")

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
        return luigi.LocalTarget(
            Path(str(os.getenv("ML_MODELS_DIR")), "trained_model", "model.pkl")
        )

    def run(self) -> None:
        """
        Run task logic. Read data from input, train model, then
        save it.
        """
        train_model.pipeline(
            input_dir=str(self.input_dir),
            input_file=str(self.input_file),
            input_test_file=str(self.input_test_file),
            input_pipeline_dir=str(self.input_pipeline_dir),
            input_pipeline_file=str(self.input_pipeline_file),
        )


if __name__ == "__main__":
    # Run all tasks, considering TrainModel requires previous tasks
    luigi.run(["TrainModel", "--local-scheduler"])
