import logging
from enum import Enum
from typing import Callable, Optional, List, Union, Literal

import giskard
import pandas as pd

# Disable Giskard logging
logging.getLogger("giskard").disabled = True


class RedTeaming:
    class SupportedModelTypes(Enum):
        TEXT_GENERATION = "text_generation"
        CLASSIFICATION = "classification"
        REGRESSION = "regression"

    ModelType = Union[SupportedModelTypes, Literal["classification", "regression", "text_generation"]]

    class ScanMetric(Enum):
        HALLUCINATION = "hallucination"
        PERFORMANCE = "performance"
        # Add other scan metrics as needed

    def __init__(self, api_key: Optional[str] = None):
        """Initialize RedTeaming instance with optional API key."""
        # self.api_key = api_key or os.getenv("GISKARD_API_KEY")
        # if not self.api_key: raise ValueError( "API key is required. Please set GISKARD_API_KEY as an environment
        # variable or pass it as a parameter.")

    def run_scan(
            self,
            model: Callable,
            model_type: ModelType,
            name: str,
            description: str,
            only: Optional[List[Union[str, ScanMetric]]] = None  # Replaced `|` with `Union`
    ) -> pd.DataFrame:
        """
        Runs red teaming on the provided model and returns a DataFrame of the results.

        :param model: The model function provided by the user.
        :param model_type: The type of the model (Enum restricted or predefined literals).
        :param name: Name of the model.
        :param description: Description of the model.
        :param only: Optional list of scan metrics to run.
        :return: A DataFrame containing the scan report.
        """
        valid_model_types = {e.value for e in self.SupportedModelTypes}.union(
            {"classification", "regression", "text_generation"}
        )
        if model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type: {model_type}. Allowed values: {valid_model_types}")

        if only:
            invalid_metrics = [metric for metric in only if
                               not isinstance(metric, (str, self.ScanMetric))]  # Adjusted type checking
            if invalid_metrics:
                raise ValueError(
                    f"Invalid scan metrics: {invalid_metrics}. Allowed values: {[e.value for e in self.ScanMetric]}")

        model_instance = giskard.Model(
            model=model,
            model_type=model_type,
            name=name,
            description=description,
            feature_names=["question"],
        )

        try:
            # Run the scan
            if only:
                report = giskard.scan(model_instance,
                                      only=[metric.value if isinstance(metric, self.ScanMetric) else metric for metric
                                            in only])
            else:
                report = giskard.scan(model_instance)
        except Exception as e:
            raise RuntimeError(f"Error occurred during model scan: {str(e)}")

        # Convert the report to a DataFrame and save it as CSV
        df = report.to_dataframe()
        df.to_csv('final_report.csv', index=False)

        return df
