from datetime import datetime
import json
import os
from typing import Dict, List, Any, Tuple

import pandas as pd
import tomli
from tqdm import tqdm

from data_generator.requirements_generator import RequirementsGenerator, RequirementsInput
from data_generator.test_case_generator import TestCaseGenerator, TestCaseInput
from evaluator import Evaluator as RequirementsEvaluator, EvaluationInput, Conversation
from utils.issue_description import get_issue_description

class RedTeaming:
    def __init__(
        self,
        model_name: str = "gpt-4-1106-preview",
        req_temperature: float = 0.7,
        test_temperature: float = 0.8,
        eval_temperature: float = 0.3
    ):
        self._load_supported_detectors()
        """
        Initialize the red teaming pipeline.
        
        Args:
            model_name: The OpenAI model to use
            req_temperature: Temperature for requirements generation
            test_temperature: Temperature for test case generation
            eval_temperature: Temperature for evaluation (lower for consistency)
        """
        # Load supported detectors configuration
        self._load_supported_detectors()
        
        # Initialize generators and evaluator
        self.req_generator = RequirementsGenerator(model_name=model_name, temperature=req_temperature)
        self.test_generator = TestCaseGenerator(model_name=model_name, temperature=test_temperature)
        self.evaluator = RequirementsEvaluator(model_name=model_name, temperature=eval_temperature)
        
    def _load_supported_detectors(self) -> None:
        """Load supported detectors from TOML configuration file."""
        config_path = os.path.join(os.path.dirname(__file__), "config", "detectors.toml")
        try:
            with open(config_path, "rb") as f:
                config = tomli.load(f)
                self.supported_detectors = set(config.get("detectors", {}).get("detector_names", []))
        except FileNotFoundError:
            print(f"Warning: Detectors configuration file not found at {config_path}")
            self.supported_detectors = set()
        except Exception as e:
            print(f"Error loading detectors configuration: {e}")
            self.supported_detectors = set()
    
    def validate_detectors(self, detectors: List[str]) -> None:
        """Validate that all provided detectors are supported.
        
        Args:
            detectors: List of detector IDs to validate
            
        Raises:
            ValueError: If any detector is not supported
        """
        unsupported = [d for d in detectors if d not in self.supported_detectors]
        if unsupported:
            raise ValueError(
                f"Unsupported detectors: {unsupported}\n"
                f"Supported detectors are: {sorted(self.supported_detectors)}"
            )
        
    def get_supported_detectors(self) -> List[str]:
        """Get the list of supported detectors."""
        return sorted(self.supported_detectors)
    
    def _get_save_path(self, description: str) -> str:
        """Generate a path for saving the final DataFrame."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a short slug from the description
        slug = description.lower()[:30].replace(" ", "_")
        return os.path.join(output_dir, f"red_teaming_{slug}_{timestamp}.csv")

    def _save_results_to_csv(self, result_df: pd.DataFrame, description: str) -> str:
        # Save DataFrame
        save_path = self._get_save_path(description)
        result_df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
        return save_path


    def _run_with_examples(self, description: str, detectors: List[str], response_model: Any, examples: List[str], num_requirements: int) -> pd.DataFrame:
        # take care of total no of requirements limit
        MAX_TOTAL_REQUIREMENTS = 5
        if len(detectors) >= MAX_TOTAL_REQUIREMENTS:
            num_requirements = 1
        elif len(detectors) * num_requirements >= MAX_TOTAL_REQUIREMENTS:
            k = 1
            while len(detectors) * k <= MAX_TOTAL_REQUIREMENTS:
                k += 1
            num_requirements = k - 1

        # generate the requirements
        requirements = []
        for detector in tqdm(detectors, desc=f"Generating requirements"):
            # issue_description = get_issue_description(detector)
            if type(detector) == str:
            # Get issue description for this detector
                issue_description = get_issue_description(detector)
            else:
                issue_description = detector.get("custom", "")
            
            # Generate requirements for this detector
            req_input = RequirementsInput(
                description=description,
                category=issue_description,
                num_requirements=num_requirements
            )
            requirement = self.req_generator.generate_requirements(req_input)
            requirements.extend(requirement)

        # Evaluate the examples against the requirements
        results = []
        failed_tests = 0
        print('-'*100)
        for example in tqdm(examples, desc=f"Evaluating examples"):
                user_message = example
                agent_response = response_model(user_message)
                
                # Evaluate the conversation
                eval_input = EvaluationInput(
                    description=description,
                    conversation=Conversation(
                        user_message=user_message,
                        agent_response=agent_response
                    ),
                    requirements=requirements
                )
                evaluation = self.evaluator.evaluate_conversation(eval_input)
                
                # Store results
                results.append({
                    "requirement": requirements,
                    "user_message": user_message,
                    "agent_response": agent_response,
                    "evaluation_passed": evaluation["eval_passed"],
                    "evaluation_reason": evaluation["reason"]
                })
                
                if not evaluation["eval_passed"]:
                    failed_tests += 1
        
        # Report results
        total_examples = len(examples)
        if failed_tests > 0:
            print(f"{failed_tests}/{total_examples} tests failed")
        else:
            print(f"All {total_examples} tests passed")
        print('-'*250)

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        save_path = self._save_results_to_csv(results_df, description)

        return results_df, save_path

    
    def _run_without_examples(self, description: str, detectors: List[str], response_model: Any, model_input_format: Dict[str, Any], num_requirements: int, num_test_cases_per_requirement: int) -> pd.DataFrame:
        results = []
        # Process each detector
        for detector in detectors:
            print('='*50)
            print(f"Running detector: {detector}")
            print('='*50)

            if type(detector) == str:
            # Get issue description for this detector
                issue_description = get_issue_description(detector)
            else:
                issue_description = detector.get("custom", "")
            
            # Generate requirements for this detector
            req_input = RequirementsInput(
                description=description,
                category=issue_description,
                num_requirements=num_requirements
            )
            requirements = self.req_generator.generate_requirements(req_input)
            
            # Process each requirement
            for r, req in enumerate(requirements):
                # Generate test cases
                test_input = TestCaseInput(
                    description=description,
                    category=issue_description,
                    requirement=req,
                    format_example=model_input_format,
                    languages=["English"],
                    num_inputs=num_test_cases_per_requirement
                )
                test_cases = self.test_generator.generate_test_cases(test_input)
                
                # Evaluate test cases
                failed_tests = 0
                with tqdm(test_cases["inputs"],
                         desc=f"Evaluating {detector} requirement {r+1}/{len(requirements)}") as pbar:
                    for test_case in pbar:
                        user_message = test_case["user_input"]
                        agent_response = response_model(user_message)
                        
                        # Evaluate the conversation
                        eval_input = EvaluationInput(
                            description=description,
                            conversation=Conversation(
                                user_message=user_message,
                                agent_response=agent_response
                            ),
                            requirements=[req]
                        )
                        evaluation = self.evaluator.evaluate_conversation(eval_input)
                        
                        # Store results
                        results.append({
                            "detector": detector,
                            "requirement": req,
                            "user_message": user_message,
                            "agent_response": agent_response,
                            "evaluation_passed": evaluation["eval_passed"],
                            "evaluation_reason": evaluation["reason"]
                        })
                        
                        if not evaluation["eval_passed"]:
                            failed_tests += 1
                
                # Report results for this requirement
                total_tests = len(test_cases["inputs"])
                if failed_tests > 0:
                    print(f"{detector} requirement {r+1}: {failed_tests}/{total_tests} tests failed")
                else:
                    print(f"{detector} requirement {r+1}: All {total_tests} tests passed")
                print('-'*100)

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        save_path = self._save_results_to_csv(results_df, description)

        return results_df, save_path

        
    def run(
        self,
        description: str,
        detectors: List[str],
        response_model: Any,
        examples: List[str] = [],
        model_input_format: Dict[str, Any] = {
            "user_input": "Hi, I am looking for job recommendations",
            "user_name": "John"
        },
        num_requirements: int = 4,
        num_test_cases_per_requirement: int = 5 # used only if examples are not provided
    ) -> pd.DataFrame:
        """
        Run the complete red teaming pipeline.
        
        Args:
            description: Description of the agent being tested
            detectors: List of detector names to test against (e.g., ["stereotypes", "harmful_content"])
            response_model: Function that takes a user message and returns the agent's response
            model_input_format: Format for test case generation
            examples: List of example inputs to test. If provided, uses these instead of generating test cases
            num_requirements: Number of requirements to generate per detector
            num_test_cases_per_requirement: Number of test cases to generate per requirement
            
        Returns:
            DataFrame containing all test results with columns:
            - requirement: The requirement being tested
            - user_message: The test input
            - agent_response: The model's response
            - evaluation_passed: Whether the response passed evaluation
            - evaluation_reason: Reason for pass/fail
        """

        # Validate detectors
        inbuild_detector = []
        for detector in detectors:
            if type(detector) == str:
                inbuild_detector.append(detector)
            elif type(detector) == dict:
                if 'custom' not in detector.keys() or len(detector.keys()) != 1:
                    raise('The custom detector must be a dictionary with only key "custom" and a string as a value')
            else:
                raise('Detector must be a string or a dictionary with only key "custom" and a string as a value')

        self.validate_detectors(inbuild_detector)
        
        if examples:
            return self._run_with_examples(description, detectors, response_model, examples, num_requirements)
            
        return self._run_without_examples(description, detectors, response_model, model_input_format, num_requirements, num_test_cases_per_requirement)


def main():
    """Example usage of the RedTeaming pipeline."""
    # Initialize pipeline with custom parameters
    def mock_response_model(text: str) -> str:
        """Mock response model that always returns a safe response."""
        return (
            "I apologize, but I cannot provide specific recommendations based on "
            "personal characteristics. Let's focus on your qualifications and goals."
        )
    
    # Initialize pipeline
    pipeline = RedTeaming(
        model_name="gpt-4-1106-preview",
        req_temperature=0.7,  # Higher temperature for varied requirements
        test_temperature=0.8,  # Higher temperature for creative test cases
        eval_temperature=0.3   # Lower temperature for consistent evaluation
    )
    
    # Define test parameters
    description = (
        "A chatbot for our recruiting platform that helps candidates find "
        "the best offers and prepare for interviews"
    )
    detectors = ["stereotypes", "harmful_content"]
    
    # Run the pipeline
    df = pipeline.run(
        description=description,
        detectors=detectors,
        response_model=mock_response_model,
        num_requirements=2,
        num_test_cases=2
    )
    
    # Display summary
    print("\nOverall Results:")
    for detector in detectors:
        detector_results = df[df["detector"] == detector]
        passed = detector_results["evaluation_passed"].sum()
        total = len(detector_results)
        print(f"{detector}: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    main()
