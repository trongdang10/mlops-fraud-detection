import logging
import pytest
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy, DataDivideStrategy
from steps.ingest_data import IngestData
from zenml.steps import step

@step
def data_test_prep_step():
    """Test the shape of the data after the data cleaning step."""
    try:
        # Ingest the data using the IngestData class
        ingest_data = IngestData()
        df = ingest_data.get_data()
        
        # Clean and preprocess the data using the DataCleaning class with DataPreprocessStrategy
        data_cleaning = DataCleaning(df, DataPreprocessStrategy())
        df = data_cleaning.handle_data()
        
        # Divide the cleaned data into training and testing sets using DataDivideStrategy
        data_divide = DataCleaning(df, DataDivideStrategy())
        X_train, X_test, y_train, y_test = data_divide.handle_data()

        # Assert the shapes of the training and testing datasets
        assert X_train.shape[1] == 11, "The number of features in the training set is not correct."  # 12 - 1 ('isFraud')
        assert X_test.shape[1] == 11, "The number of features in the testing set is not correct."
        
        # Check the size of training and testing sets based on an 80-20 split
        total_samples = df.shape[0]
        assert X_train.shape[0] == int(0.7 * total_samples), "The number of training samples is not correct."
        assert X_test.shape[0] == total_samples - X_train.shape[0], "The number of testing samples is not correct."
        
        logging.info("Data Shape Assertion test passed.")  # Log that the shape assertion test passed
    except Exception as e:
        pytest.fail(f"Data shape test failed: {e}")  # Fail the test and log the exception if an error occurs

@step
def check_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Test if there is any data leakage."""
    try:
        # Check if there is any intersection between the training and testing datasets
        assert len(X_train.index.intersection(X_test.index)) == 0, "There is data leakage."
        
        logging.info("Data Leakage test passed.")  # Log that the data leakage test passed
    except Exception as e:
        pytest.fail(f"Data leakage test failed: {e}")  # Fail the test and log the exception if an error occurs


