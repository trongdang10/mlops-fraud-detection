import logging

import mlflow
import pandas as pd
from src.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import ClassifierMixin
from zenml import step
from zenml.client import Client

from .config import ModelNameConfig

# Get the active experiment tracker from ZenML's active stack
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name) #The @step decorator integrates the function with ZenML’s pipeline management and tracking system.
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> ClassifierMixin:
    """
    Trains a model based on the provided configuration and data.

    Args:
        x_train: Training data features as a DataFrame.
        x_test: Testing data features as a DataFrame.
        y_train: Training data target as a Series.
        y_test: Testing data target as a Series.
        config: Configuration object specifying which model to train and other settings.

    Returns:
        trained_model: A trained classifier model.
    """
    try:
        model = None  # Initialize the model variable
        tuner = None  # Initialize the tuner variable

        # Select the model based on the configuration
        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()  # Enable MLflow autologging for LightGBM
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()  # Enable MLflow autologging for sklearn models
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()  # Enable MLflow autologging for XGBoost
            model = XGBoostModel()
        elif config.model_name == "logistic_regression":
            mlflow.sklearn.autolog()  # Enable MLflow autologging for sklearn models
            model = LogisticRegressionModel()
        else:
            raise ValueError("Model name not supported")  # Raise an error if the model name is not supported

        # Initialize the hyperparameter tuner with the selected model and data
        tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)

        # Perform hyperparameter tuning if fine_tuning is enabled in the configuration
        if config.fine_tuning:
            best_params = tuner.optimize()  # Optimize hyperparameters and get the best parameters
            trained_model = model.train(X_train, y_train, **best_params)  # Train the model with the best parameters
        else:
            trained_model = model.train(X_train, y_train)  # Train the model with default parameters
        
        return trained_model  # Return the trained model
    except Exception as e:
        logging.error(e)  # Log the error if any exception occurs
        raise e  # Raise the exception to ensure it is not suppressed
