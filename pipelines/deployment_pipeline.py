import json  # Import json module for handling JSON data

import os  # Import os module for interacting with the operating system

import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation

# Import custom materializer and steps
from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_data
from steps.evaluate_model import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model

# Import ZenML modules
from zenml import pipeline, step  # Import pipeline and step decorators from ZenML
from zenml.config import DockerSettings  # Import DockerSettings for configuring Docker in ZenML
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT  # Import default service timeout constant
from zenml.integrations.constants import MLFLOW, TENSORFLOW  # Import integration constants
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)  # Import MLFlowModelDeployer for deploying models with MLflow
from zenml.integrations.mlflow.services import MLFlowDeploymentService  # Import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step  # Import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output  # Import BaseParameters and Output for defining step parameters

# Import utility function
from .utils import get_data_for_test

# Configure Docker settings for the pipeline, specifying MLflow integration
docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()  # Fetch the latest data
    return data  # Return the data as a string

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0.9  # Minimum accuracy threshold for deployment

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """
    Implements a simple model deployment trigger that looks at the input model accuracy
    and decides if it is good enough to deploy.

    Args:
        accuracy: The accuracy of the model.
        config: Configuration parameters for deployment.

    Returns:
        bool: True if the model accuracy is above the threshold, False otherwise.
    """
    return accuracy > config.min_accuracy  # Check if accuracy meets the threshold

# class MLFlowDeploymentLoaderStepParameters(BaseParameters):
#     """
#     MLflow deployment getter parameters.

#     Attributes:
#         pipeline_name: Name of the pipeline that deployed the MLflow prediction server.
#         step_name: Name of the step that deployed the MLflow prediction server.
#         running: Flag to return only a running service.
#         model_name: Name of the model that is deployed.
#     """
#     pipeline_name: str
#     step_name: str
#     running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """
    Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: Name of the pipeline that deployed the MLflow prediction server.
        pipeline_step_name: Name of the step that deployed the MLflow prediction server.
        running: Flag to return only a running service.
        model_name: Name of the model that is deployed.

    Returns:
        MLFlowDeploymentService: The MLflow deployment service.
    """
    # Get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Fetch existing services with the same pipeline name, step name, and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    # Raise an error if no running service is found
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]  # Return the first found service

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """
    Run an inference request against a prediction service.

    Args:
        service: The MLflow deployment service.
        data: The data to predict on.

    Returns:
        np.ndarray: The predictions from the model.
    """
    service.start(timeout=10)  # Start the service (no-op if already started)
    data = json.loads(data)  # Load the data from JSON
    data.pop("columns")  # Remove the 'columns' key
    data.pop("index")  # Remove the 'index' key

    # Define columns for DataFrame
    columns_for_df = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)  # Create DataFrame from data
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))  # Convert DataFrame to JSON list
    data = np.array(json_list)  # Convert JSON list to numpy array
    prediction = service.predict(data)  # Get predictions from the service
    return prediction  # Return the predictions

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """
    Continuous deployment pipeline to ingest, clean, train, evaluate, and deploy a model.

    Args:
        min_accuracy: Minimum accuracy threshold for deployment.
        workers: Number of workers for deployment.
        timeout: Timeout for starting/stopping services.
    """
    # Link all the steps artifacts together
    df = ingest_data()  # Ingest data
    X_train, X_test, y_train, y_test = clean_data(df)  # Clean and preprocess data
    model = train_model(X_train, X_test, y_train, y_test)  # Train the model
    accuracy, precision, recall ,f1 = evaluation(model, X_test, y_test)  # Evaluate the model
    deployment_decision = deployment_trigger(accuracy=f1)  # Trigger deployment decision
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )  # Deploy the model if criteria are met

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    """
    Inference pipeline to load data, fetch model service, and make predictions.

    Args:
        pipeline_name: Name of the pipeline that deployed the model.
        pipeline_step_name: Name of the step that deployed the model.
    """
    # Link all the steps artifacts together
    batch_data = dynamic_importer()  # Load batch data
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )  # Load the deployed model service
    prediction = predictor(service=model_deployment_service, data=batch_data)  # Make predictions
    return prediction
