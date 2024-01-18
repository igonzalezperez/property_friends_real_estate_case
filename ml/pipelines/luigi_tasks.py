import luigi
import pandas as pd
from pathlib import Path
import luigi
import pandas as pd
from pathlib import Path
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
import luigi
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load
from loguru import logger
import os

raw_data_folder = "ml/data/raw"
interim_data_folder = "ml/data/interim"
models_folder = "ml/models"

input_data_file = "train.csv"
preproc_pipeline_file = "preproc_pipeline.joblib"
model_file = "model.joblib"


class MakeDataset(luigi.Task):
    input_dir = luigi.Parameter(default=raw_data_folder)
    output_dir = luigi.Parameter(default=interim_data_folder)

    def output(self):
        return luigi.LocalTarget(Path(self.output_dir, input_data_file))

    def run(self):
        data_in_path = Path(self.input_dir, input_data_file)
        data_out_path = self.output().path

        logger.info(f"Reading data from {data_in_path}")
        data = pd.read_csv(data_in_path)

        logger.info(f"Saving data to {data_out_path}")
        data.to_csv(data_out_path, index=False)
        logger.success("MakeDataset process ran successfully")


class BuildFeatures(luigi.Task):
    input_data_dir = luigi.Parameter(default=interim_data_folder)
    output_model_dir = luigi.Parameter(default=models_folder)

    def requires(self):
        return MakeDataset()

    def output(self):
        return luigi.LocalTarget(Path(self.output_model_dir, preproc_pipeline_file))

    def run(self):
        data_path = Path(self.input_data_dir, input_data_file)

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


class TrainModel(luigi.Task):
    input_data_dir = luigi.Parameter(default=interim_data_folder)
    input_model_dir = luigi.Parameter(default=models_folder)
    output_model_dir = luigi.Parameter(default=models_folder)

    def requires(self):
        return MakeDataset()

    def output(self):
        return luigi.LocalTarget(Path(self.output_model_dir, model_file))

    def run(self):
        data_path = Path(self.input_data_dir, input_data_file)
        preproc_path = Path(self.input_model_dir, preproc_pipeline_file)

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
    luigi.run(["TrainModel", "--local-scheduler"])
