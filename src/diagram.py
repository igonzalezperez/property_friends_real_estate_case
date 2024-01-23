"""
Generate diagram of the Project
"""
from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.aws.ml import SagemakerModel
from diagrams.aws.storage import S3
from diagrams.onprem.database import PostgreSQL
from diagrams.programming.framework import FastAPI
from diagrams.onprem.mlops import Mlflow
from diagrams.aws.mobile import APIGateway
from diagrams.aws.storage import SimpleStorageServiceS3Bucket
from diagrams.aws.analytics import DataPipeline
from pathlib import Path


def main() -> None:
    """
    Creates and saves a diagram of the project's architecture using the
    diagrams library.
    """
    with Diagram(
        "Project Architecture",
        show=False,
        direction="LR",
        filename="assets/project_architecture",
        outformat="png",
    ):
        postgres_db = PostgreSQL("PostgreSQL DB")

        csv_data = S3("Data Source [.csv]")
        mlruns = SimpleStorageServiceS3Bucket("MLRuns")
        ml_model = SagemakerModel("Model Artifact [.pkl]")
        predict_logs = SimpleStorageServiceS3Bucket("Predict Logs [.json]")
        pipeline_params = SimpleStorageServiceS3Bucket("Pipeline Params [.yaml]")

        csv_data - Edge(style="dashed") >> postgres_db

        with Cluster("Pipeline Orchestration"):
            # If the path isn't absolute this won't load
            luigi_task = Custom(
                "Luigi Tasks", str(Path("assets", "luigi.png").resolve())
            )
            csv_data >> luigi_task
            postgres_db >> luigi_task
            with Cluster("Pipelines"):
                make_data = DataPipeline("Make dataset")
                build_features = DataPipeline("Build Features")
                train = Mlflow("Train with MLFlow")
                make_data >> build_features >> train
        luigi_task >> make_data
        train >> mlruns
        train >> ml_model

        with Cluster("App"):
            app = FastAPI("FastAPI")
            with Cluster("Endpoints"):
                predict = APIGateway("/predict")
                health = APIGateway("/health")
                health - Edge(style="dashed") >> predict
                predict_logs_endpoint = APIGateway("/predict-logs")
                model_runs = APIGateway("/model-runs")
                ml_model >> predict
        pipeline_params >> build_features
        pipeline_params >> train
        predict >> predict_logs
        predict_logs >> predict_logs_endpoint
        mlruns >> model_runs
        api_key_storage = SimpleStorageServiceS3Bucket("Valid API keys [.json]")
        app << api_key_storage


if __name__ == "__main__":
    main()
