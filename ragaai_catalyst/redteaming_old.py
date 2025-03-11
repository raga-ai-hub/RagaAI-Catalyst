# import logging
# import os
# from typing import Callable, Optional

# import giskard as scanner
# import pandas as pd

# logging.getLogger('giskard.core').disabled = True
# logging.getLogger('giskard.scanner.logger').disabled = True
# logging.getLogger('giskard.models.automodel').disabled = True
# logging.getLogger('giskard.datasets.base').disabled = True
# logging.getLogger('giskard.utils.logging_utils').disabled = True


# class RedTeaming:

#     def __init__(self,
#                  provider: Optional[str] = "openai",
#                  model: Optional[str] = None,
#                  api_key: Optional[str] = None,
#                  api_base: Optional[str] = None,
#                  api_version: Optional[str] = None):
#         self.provider = provider.lower()
#         self.model = model
#         if not self.provider:
#             raise ValueError("Model configuration must be provided with a valid provider and model.")
#         if self.provider == "openai":
#             if api_key is not None:
#                 os.environ["OPENAI_API_KEY"] = api_key
#             if os.getenv("OPENAI_API_KEY") is None:
#                 raise ValueError("API key must be provided for OpenAI.")
#         elif self.provider == "gemini":
#             if api_key is not None:
#                 os.environ["GEMINI_API_KEY"] = api_key
#             if os.getenv("GEMINI_API_KEY") is None:
#                 raise ValueError("API key must be provided for Gemini.")
#         elif self.provider == "azure":
#             if api_key is not None:
#                 os.environ["AZURE_API_KEY"] = api_key
#             if api_base is not None:
#                 os.environ["AZURE_API_BASE"] = api_base
#             if api_version is not None:
#                 os.environ["AZURE_API_VERSION"] = api_version
#             if os.getenv("AZURE_API_KEY") is None:
#                 raise ValueError("API key must be provided for Azure.")
#             if os.getenv("AZURE_API_BASE") is None:
#                 raise ValueError("API base must be provided for Azure.")
#             if os.getenv("AZURE_API_VERSION") is None:
#                 raise ValueError("API version must be provided for Azure.")
#         else:
#             raise ValueError(f"Provider is not recognized.")

#     def run_scan(
#             self,
#             model: Callable,
#             evaluators: Optional[list] = None,
#             save_report: bool = True
#     ) -> pd.DataFrame:
#         """
#         Runs red teaming on the provided model and returns a DataFrame of the results.

#         :param model: The model function provided by the user (can be sync or async).
#         :param evaluators: Optional list of scan metrics to run.
#         :param save_report: Boolean flag indicating whether to save the scan report as a CSV file.
#         :return: A DataFrame containing the scan report.
#         """
#         import asyncio
#         import inspect

#         self.set_scanning_model(self.provider, self.model)

#         supported_evaluators = self.get_supported_evaluators()
#         if evaluators:
#             if isinstance(evaluators, str):
#                 evaluators = [evaluators]
#             invalid_evaluators = [evaluator for evaluator in evaluators if evaluator not in supported_evaluators]
#             if invalid_evaluators:
#                 raise ValueError(f"Invalid evaluators: {invalid_evaluators}. "
#                                  f"Allowed evaluators: {supported_evaluators}.")

#         # Handle async model functions by wrapping them in a sync function
#         if inspect.iscoroutinefunction(model):
#             def sync_wrapper(*args, **kwargs):
#                 try:
#                     # Try to get the current event loop
#                     loop = asyncio.get_event_loop()
#                 except RuntimeError:
#                     # If no event loop exists (e.g., in Jupyter), create a new one
#                     loop = asyncio.new_event_loop()
#                     asyncio.set_event_loop(loop)

#                 try:
#                     # Handle both IPython and regular Python environments
#                     import nest_asyncio
#                     nest_asyncio.apply()
#                 except ImportError:
#                     pass  # nest_asyncio not available, continue without it

#                 return loop.run_until_complete(model(*args, **kwargs))
#             wrapped_model = sync_wrapper
#         else:
#             wrapped_model = model

#         model_instance = scanner.Model(
#             model=wrapped_model,
#             model_type="text_generation",
#             name="RagaAI's Scan",
#             description="RagaAI's RedTeaming Scan",
#             feature_names=["question"],
#         )

#         try:
#             report = scanner.scan(model_instance, only=evaluators, raise_exceptions=True) if evaluators \
#                      else scanner.scan(model_instance, raise_exceptions=True)
#         except Exception as e:
#             raise RuntimeError(f"Error occurred during model scan: {str(e)}")

#         report_df = report.to_dataframe()

#         if save_report:
#             report_df.to_csv("raga-ai_red-teaming_scan.csv", index=False)

#         return report_df

#     def get_supported_evaluators(self):
#         """Contains tags corresponding to the 'llm' and 'robustness' directories in the giskard > scanner library"""
#         return {'control_chars_injection',
#                 'discrimination',
#                 'ethical_bias',
#                 'ethics',
#                 'faithfulness',
#                 'generative',
#                 'hallucination',
#                 'harmfulness',
#                 'implausible_output',
#                 'information_disclosure',
#                 'jailbreak',
#                 'llm',
#                 'llm_harmful_content',
#                 'llm_stereotypes_detector',
#                 'misinformation',
#                 'output_formatting',
#                 'prompt_injection',
#                 'robustness',
#                 'stereotypes',
#                 'sycophancy',
#                 'text_generation',
#                 'text_perturbation'}

#     def set_scanning_model(self, provider, model=None):
#         """
#         Sets the LLM model for Giskard based on the provider.

#         :param provider: The LLM provider (e.g., "openai", "gemini", "azure").
#         :param model: The specific model name to use (optional).
#         :raises ValueError: If the provider is "azure" and no model is provided.
#         """
#         default_models = {
#             "openai": "gpt-4o",
#             "gemini": "gemini-1.5-pro"
#         }

#         if provider == "azure" and model is None:
#             raise ValueError("Model must be provided for Azure.")

#         selected_model = model if model is not None else default_models.get(provider)

#         if selected_model is None:
#             raise ValueError(f"Unsupported provider: {provider}")

#         scanner.llm.set_llm_model(selected_model)
