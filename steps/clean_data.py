import logging  # Import logging module to log information and errors
from typing import Tuple  # Import Tuple for type annotations

import pandas as pd  # Import pandas for data manipulation
from src.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)  # Import data cleaning strategies and classes from src.data_cleaning
from typing_extensions import Annotated  # Import Annotated for detailed type annotations

from zenml import step  # Import step decorator from ZenML to define pipeline steps

@step
def clean_data(
    df: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Data cleaning function that preprocesses the data and divides it into train and test sets.

    Args:
        data: pd.DataFrame - The raw data to be cleaned and divided.

    Returns:
        Tuple containing:
            x_train: pd.DataFrame - The training features.
            x_test: pd.DataFrame - The testing features.
            y_train: pd.Series - The training labels.
            y_test: pd.Series - The testing labels.
    """
    try:
        # Initialize the preprocessing strategy and apply it to the data
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        # Initialize the divide strategy and apply it to the preprocessed data
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        y_train = y_train.squeeze()  # Converts to Series if it's a DataFrame with a single column
        y_test = y_test.squeeze()    # Converts to Series if it's a DataFrame with a single column
        logging.info("Data cleaning completed")
        # Return the divided datasets
        return X_train, X_test, y_train, y_test
    except Exception as e:
        # Log any exceptions that occur during data cleaning and division
        logging.error("Error in cleaning data: {}".format(e))
        # Re-raise the exception to ensure the error is not silently ignored
        raise e
