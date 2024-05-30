from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """Model Configurations
    
    Attributes:
        model_name (str): The name of the model to train. Default is "lightgbm".
        fine_tuning (bool): Whether to perform hyperparameter fine-tuning. Default is False.
    """
    model_name: str = "lightgbm"
    fine_tuning: bool = False
