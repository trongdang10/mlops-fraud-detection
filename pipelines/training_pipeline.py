from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluate_model import evaluation
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    """
    ZenML pipeline to train a model.
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        accuracy: float
        f1: float
    """
    # Ingest data using the provided ingest_data step function
    df = ingest_data(data_path)
    
    # Clean and preprocess the data using the clean_data step function
    # The clean_data function is expected to return the training and testing datasets
    X_train, X_test, y_train, y_test = clean_data(df)
    
    # Train the model using the model_train step function
    # The model_train function is expected to return the trained model
    model = train_model(X_train, X_test, y_train, y_test)
    
    # Evaluate the model using the evaluation step function
    # The evaluation function is expected to return evaluation metrics such as accuracy and F1 score
    accuracy, precision, recall ,f1 = evaluation(model, X_test, y_test)
