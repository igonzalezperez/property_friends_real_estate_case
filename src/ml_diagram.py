"""
Generate diagram of the Project
"""
from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import DataPipeline
from diagrams.aws.ml import SagemakerModel
from diagrams.aws.storage import S3, SimpleStorageServiceS3Bucket
from diagrams.custom import Custom
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.mlops import Mlflow


def main() -> None:
    """
    Creates and saves a diagram of the ML architecture using the
    diagrams library.
    """
    with Diagram(
        "DB Container Architecture",
        show=False,
        direction="LR",
        filename="assets/ml_architecture",
        outformat="png",
        graph_attr={"dpi": "300"},
    ):
        with Cluster("DB Container"):
            postgres_db = PostgreSQL("PostgreSQL DB")
            csv_data = S3("Data Source [.csv]")
            with Cluster("ML Artifacts"):
                mlruns = SimpleStorageServiceS3Bucket("MLRuns")
                ml_model = SagemakerModel("Model Artifact [.pkl]")

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
            pipeline_params = SimpleStorageServiceS3Bucket("Pipeline Params [.yaml]")
            pipeline_params >> build_features
            pipeline_params >> train


if __name__ == "__main__":
    main()
