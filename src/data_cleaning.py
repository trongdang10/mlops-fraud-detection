import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for preprocessing data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            data['isFraud'] = data['isFraud'].astype('object') #convert type of isFraud from int to object

            data = data.loc[data['type'].isin(['CASH_OUT', 'TRANSFER']),:] # filter out type of transaction that has fraudulent transactions

            data = data.loc[data['amount'] > 0,:] #filter out account that has 0 transaction
            
            data = pd.get_dummies(data, columns=['type'], prefix=['type']) #convert CASH_OUT and TRANSFER into dummy var

            cols_to_drop = ["nameOrig", "nameDest", "isFlaggedFraud"] #drop column
            data = data.drop(cols_to_drop, axis=1)

            # Encode the 'isFraud' column if it's of object type
            if data['isFraud'].dtype == 'object':
                label_encoder = LabelEncoder()
                data['isFraud'] = label_encoder.fit_transform(data['isFraud'])

            return data
        except Exception as e:
            logging.error("Error in preprocessing data; {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:

            X = data.drop(["isFraud"], axis=1)
            y = data["isFraud"]

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        # Initializes the DataCleaning class with a specific strategy.
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        # Handle data based on the provided strategy
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
