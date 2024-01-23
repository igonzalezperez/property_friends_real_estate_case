"""
Generate diagram of the Project
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.ml import SagemakerModel
from diagrams.aws.mobile import APIGateway
from diagrams.aws.storage import SimpleStorageServiceS3Bucket
from diagrams.programming.framework import FastAPI


def main() -> None:
    """
    Creates and saves a diagram of the app's architecture using the
    diagrams library.
    """
    with Diagram(
        "App Container Architecture",
        show=False,
        direction="LR",
        filename="assets/app_architecture",
        outformat="png",
        graph_attr={"dpi": "300"},
    ):
        with Cluster("Project"):
            with Cluster("ML Artifacts"):
                mlruns = SimpleStorageServiceS3Bucket("MLRuns")
                ml_model = SagemakerModel("Model Artifact [.pkl]")
            api_key_storage = SimpleStorageServiceS3Bucket("Valid API keys [.json]")

            with Cluster("App"):
                app = FastAPI("FastAPI")
                with Cluster("Endpoints"):
                    predict = APIGateway("/predict")
                    health = APIGateway("/health")
                    health - Edge(style="dashed") >> predict
                    predict_logs_endpoint = APIGateway("/predict-logs")
                    model_runs = APIGateway("/model-runs")
                    ml_model >> predict
            predict_logs = SimpleStorageServiceS3Bucket("Predict Logs [.json]")
            predict >> predict_logs
            predict_logs >> predict_logs_endpoint
            mlruns >> model_runs
            app << api_key_storage


if __name__ == "__main__":
    main()
