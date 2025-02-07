import giskard
import pandas as pd
import logging
from enum import Enum
from typing import Callable, Optional, List, Union, Literal

# Disable Giskard logging
logging.getLogger("giskard").setLevel(logging.CRITICAL)


class SupportedModelTypes(Enum):
    TEXT_GENERATION = "text_generation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


ModelType = Union[SupportedModelTypes, Literal["classification", "regression", "text_generation"]]


class ScanMetric(Enum):
    HALLUCINATION = "hallucination"
    PERFORMANCE = "performance"
    # Other metrics to be added here


def analyze_model_risks(
        model: Callable,
        model_type: ModelType,
        name: str,
        description: str,
        only: Optional[List[ScanMetric]] = None
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
    valid_model_types = {e.value for e in SupportedModelTypes} | {"classification", "regression", "text_generation"}
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type: {model_type}. Allowed values: {valid_model_types}")

    if only:
        invalid_metrics = [metric for metric in only if not isinstance(metric, ScanMetric)]
        if invalid_metrics:
            raise ValueError(
                f"Invalid scan metrics: {invalid_metrics}. Allowed values: {[e.value for e in ScanMetric]}")

    model_instance = giskard.Model(
        model=model,
        model_type=model_type,
        name=name,
        description=description,
        feature_names=["question"],
    )

    # Run the scan
    if only:
        report = giskard.scan(model_instance, only=[metric.value for metric in only])
    else:
        report = giskard.scan(model_instance)

    # Convert the report to a DataFrame and save it as CSV
    df = report.to_dataframe()
    df.to_csv('final_report.csv', index=False)

    return df
