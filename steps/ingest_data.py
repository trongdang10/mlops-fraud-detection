import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Reads data from a CSV file and returns it as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the ingested data
        """
        # Read data from the CSV file into a DataFrame
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    ZenML step for ingesting data.

    This function creates an instance of IngestData, uses it to read data from a CSV file,
    and returns the resulting DataFrame. If an error occurs, it logs the error and raises it.

    Args:
        None

    Returns:
        pd.DataFrame: DataFrame containing the ingested data
    """
    try:
        # Create an instance of the IngestData class
        ingest_data = IngestData(data_path)
        
        # Use the instance to get data and return the DataFrame
        df = ingest_data.get_data()
        return df
    except Exception as e:
        # Log any exceptions that occur during data ingestion
        logging.error(f"Error while ingesting data: {e}")
        # Re-raise the exception to ensure the error is not silently ignored
        raise e
