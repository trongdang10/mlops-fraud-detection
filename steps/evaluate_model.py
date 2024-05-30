import logging

import mlflow
import numpy as np
import pandas as pd
from src.evaluation import Accuracy, Precision, Recall, F1
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from typing import Tuple

# Get the active experiment tracker from ZenML's active stack
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[
            Annotated[float, "accuracy"], 
           Annotated[float, "precision"], 
           Annotated[float, "recall"], 
           Annotated[float, "f1"]
           ]:
    """
    Evaluate the performance of a classification model.

    Args:
        model: ClassifierMixin - The trained classification model.
        X_test: pd.DataFrame - Test data features.
        y_test: pd.Series - True labels for the test data.

    Returns:
        Tuple containing accuracy, precision, recall, and F1 score.
    """
    try:
        # Make predictions using the model
        prediction = model.predict(X_test)

        # Calculate and log accuracy
        accuracy_class = Accuracy()  # Instantiate the Accuracy evaluation strategy
        accuracy = accuracy_class.calculate_score(y_test.to_numpy(), prediction)  # Calculate accuracy
        mlflow.log_metric("accuracy", accuracy)  # Log accuracy to MLflow

        # Calculate and log precision
        precision_class = Precision()  # Instantiate the Precision evaluation strategy
        precision = precision_class.calculate_score(y_test.to_numpy(), prediction)  # Calculate precision
        mlflow.log_metric("precision", precision)  # Log precision to MLflow

        # Calculate and log recall
        recall_class = Recall()  # Instantiate the Recall evaluation strategy
        recall = recall_class.calculate_score(y_test.to_numpy(), prediction)  # Calculate recall
        mlflow.log_metric("recall", recall)  # Log recall to MLflow

        # Calculate and log F1 score
        f1_class = F1()  # Instantiate the F1 evaluation strategy
        f1 = f1_class.calculate_score(y_test.to_numpy(), prediction)  # Calculate F1 score
        mlflow.log_metric("f1", f1)  # Log F1 score to MLflow

        # Return the calculated metrics
        return accuracy, precision, recall, f1
    except Exception as e:
        logging.error(e)  # Log the error
        raise e  # Raise the exception to ensure it is not suppressed
