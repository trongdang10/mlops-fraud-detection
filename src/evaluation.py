import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: predicted labels
        Returns:
            None
        """
        pass


class Accuracy(Evaluation):
    """
    Evaluation strategy that uses Accuracy.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            y_true: np.ndarray - True labels.
            y_pred: np.ndarray - Predicted labels.
        
        Returns:
            accuracy: float - Calculated accuracy score.
        """
        try:
            logging.info("Entered the calculate_score method of the Accuracy class")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("The accuracy score value is: " + str(accuracy))
            return accuracy
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the Accuracy class. Exception message:  "
                + str(e)
            )
            raise e


class Precision(Evaluation):
    """
    Evaluation strategy that uses Precision.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the Precision class")
            precision = precision_score(y_true, y_pred)
            logging.info("The precision score value is: " + str(precision))
            return precision
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the Precision class. Exception message:  "
                + str(e)
            )
            raise e


class Recall(Evaluation):
    """
    Evaluation strategy that uses Recall.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the Recall class")
            recall = recall_score(y_true, y_pred)
            logging.info("The recall score value is: " + str(recall))
            return recall
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the Recall class. Exception message:  "
                + str(e)
            )
            raise e


class F1(Evaluation):
    """
    Evaluation strategy that uses F1 Score.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the F1 class")
            f1 = f1_score(y_true, y_pred)
            logging.info("The F1 score value is: " + str(f1))
            return f1
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the F1 class. Exception message:  "
                + str(e)
            )
            raise e