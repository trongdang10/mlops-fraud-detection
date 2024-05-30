import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.

        Args:
            X_train: Training data
            y_train: Target data
        """
        pass

    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Target data
            X_test: Testing data
            y_test: Testing target
        """
        pass


class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        clf = RandomForestClassifier(**kwargs)
        clf.fit(X_train, y_train)
        return clf


    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        
        clf = self.train(X_train, y_train, 
                         n_estimators=n_estimators, 
                         max_depth=max_depth, 
                         min_samples_split=min_samples_split, 
                         min_samples_leaf=min_samples_leaf)
        y_pred = clf.predict(X_test)
        accuracy =accuracy_score(y_test, y_pred)
        return accuracy


class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface using LGBMClassifier.
    """

    def train(self, X_train, y_train, **kwargs):
        clf = LGBMClassifier(**kwargs)
        clf.fit(X_train, y_train)
        return clf

    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        
        # Train the model with the suggested parameters
        clf = self.train(X_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        
        # predict and Evaluate
        y_pred = clf.predict(X_test)
        accuracy =accuracy_score(y_test, y_pred)
        return accuracy


class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        clf = xgb.XGBClassifier(**kwargs)
        clf.fit(X_train, y_train)
        return clf

    def optimize(self, trial,  X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        
        # Train the model with the suggested parameters
        clf = self.train(X_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        
        # predict and Evaluate
        y_pred = clf.predict(X_test)
        accuracy =accuracy_score(y_test, y_pred)
        return accuracy


class LogisticRegressionModel(Model):
    """
    LogisticRegressionModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        clf = LogisticRegression(**kwargs)
        clf.fit(X_train, y_train)
        return clf

    # For logistic regression, we can return the accuracy score as the performance metric
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        clf = self.train(X_train, y_train)
        # predict and Evaluate
        y_pred = clf.predict(X_test)
        accuracy =accuracy_score(y_test, y_pred)
        return accuracy

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params
