from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":

    train_pipeline(data_path = "./data/fraud_detect.csv")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
