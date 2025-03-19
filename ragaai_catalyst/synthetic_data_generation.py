import os
import ast
import csv
import json
import random
import pypdf
import markdown
import pandas as pd
from tqdm import tqdm

import openai
import tiktoken
import litellm
from groq import Groq
from litellm import completion

from .internal_api_completion import api_completion as internal_api_completion
from .proxy_call import api_completion as proxy_api_completion

from typing import Optional, List, Dict, Any

import logging

logger = logging.getLogger(__name__)

class SyntheticDataGeneration:
    """
    A class for generating synthetic data using various AI models and processing different document types.
    """

    def __init__(self):
        """
        Initialize the SyntheticDataGeneration class with API clients for Groq, Gemini, and OpenAI.
        """

    def generate_qna(self, text, question_type="simple", n=5, model_config=dict(), api_key=None, **kwargs):
        """
        Generate questions based on the given text using the specified model and provider.
        Uses batch processing for larger values of n to maintain response quality.

        Args:
            text (str): The input text to generate questions from.
            question_type (str): The type of questions to generate ('simple', 'mcq', or 'complex').
            n (int): The number of question/answer pairs to generate.
            model_config (dict): Configuration for the model including provider and model name.
            api_key (str, optional): The API key for the selected provider.
            **kwargs: Additional keyword arguments.

        Returns:
            pandas.DataFrame: A DataFrame containing exactly n generated questions and answers.

        Raises:
            ValueError: If an invalid provider is specified or API key is missing.
        """
        text_validity = self.validate_input(text)
        if text_validity:
            raise ValueError(text_validity)

        BATCH_SIZE = 5  # Optimal batch size for maintaining response quality
        provider = model_config.get("provider")
        model = model_config.get("model")
        api_base = model_config.get("api_base")
        api_version = model_config.get("api_version")

        # Initialize the appropriate client based on provider
        self._initialize_client(provider, api_key, api_base, api_version, internal_llm_proxy=kwargs.get("internal_llm_proxy", None))

        # Initialize progress bar
        pbar = tqdm(total=n, desc="Generating QA pairs")
        
        # Initial generation phase
        num_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
        all_responses = []

        FAILURE_CASES = [
            "Invalid API key provided",
            "No connection adapters", 
            "Required API Keys are not set",
            "litellm.BadRequestError",
            "litellm.AuthenticationError", 
            "Max retries exceeded"
            ]
        
        for _ in range(num_batches):
            current_batch_size = min(BATCH_SIZE, n - len(all_responses))
            if current_batch_size <= 0:
                break
                
            try:
                system_message = self._get_system_message(question_type, current_batch_size)
                if "internal_llm_proxy" in kwargs:
                    batch_df = self._generate_internal_response(text, system_message, model_config, kwargs)
                else:
                    batch_df = self._generate_batch_response(text, system_message, provider, model_config, api_key, api_base)
                
                if not batch_df.empty and len(batch_df) > 0:
                    all_responses.extend(batch_df.to_dict('records'))
                    pbar.update(len(batch_df))
                    
            except Exception as e:
                print(f"Batch generation failed:{str(e)}")

                if any(error in str(e) for error in FAILURE_CASES):
                    raise Exception(f"{e}")

                else:
                    if "'utf-8' codec can't encode characters" in str(e):
                        print('Encountered non utf charactes, retrying with processed text')
                        text = str(text.encode('utf-8',errors='ignore'))
                    print(f"Retrying...")
                    continue
        
        
        # Convert to DataFrame and remove duplicates
        result_df = pd.DataFrame(all_responses)
        result_df = result_df.drop_duplicates(subset=['Question'])
        
        # Replenish phase - generate additional questions if needed due to duplicates
        while (len(result_df) < n) and ((len(result_df) >= 1)):
            questions_needed = n - len(result_df)
            try:
                system_message = self._get_system_message(question_type, questions_needed)
                
                if "internal_llm_proxy" in kwargs:
                    additional_df = self._generate_internal_response(text, system_message, model_config, kwargs)
                else:
                    additional_df = self._generate_batch_response(text, system_message, provider, model_config, api_key, api_base)
                
                if not additional_df.empty and len(additional_df) > 0:
                    # Only add questions that aren't already in result_df
                    new_questions = additional_df[~additional_df['Question'].isin(result_df['Question'])]
                    if not new_questions.empty:
                        result_df = pd.concat([result_df, new_questions], ignore_index=True)
                        result_df = result_df.drop_duplicates(subset=['Question'])
                        pbar.update(len(new_questions))
                    
            except Exception as e:
                print(f"Replenishment generation failed")

                if any(error in str(e) for error in FAILURE_CASES):
                    raise Exception(f"{e}")
                
                else:
                    print("An unexpected error occurred. Retrying...")
                    continue
        
        pbar.close()
        
        # Ensure exactly n rows and reset index starting from 1
        final_df = result_df.head(n)
        final_df.index = range(1, len(final_df) + 1)
        
        return final_df

    def _initialize_client(self, provider, api_key, api_base=None, api_version=None, internal_llm_proxy=None):
        """Initialize the appropriate client based on provider."""
        if not provider:
            raise ValueError("Model configuration must be provided with a valid provider and model.")

        if provider == "groq":
            if api_key is None and os.getenv("GROQ_API_KEY") is None:
                raise ValueError("API key must be provided for Groq.")
            self.groq_client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        
        elif provider == "gemini":
            if api_key is None and os.getenv("GEMINI_API_KEY") is None and api_base is None and internal_llm_proxy is None:
                raise ValueError("API key must be provided for Gemini.")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            # genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        
        elif provider == "openai":
            if api_key is None and os.getenv("OPENAI_API_KEY") is None and internal_llm_proxy is None:
                raise ValueError("API key must be provided for OpenAI.")
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")

        elif provider == "azure":
            if api_key is None and os.getenv("AZURE_API_KEY") is None and internal_llm_proxy is None:
                raise ValueError("API key must be provided for Azure.")
            litellm.api_key = api_key or os.getenv("AZURE_API_KEY")
            if api_base is None and os.getenv("AZURE_API_BASE") is None and internal_llm_proxy is None:
                raise ValueError("API Base must be provided for Azure.")
            litellm.api_base = api_base or os.getenv("AZURE_API_BASE")
            if api_version is None and os.getenv("AZURE_API_VERSION") is None and internal_llm_proxy is None:
                raise ValueError("API version must be provided for Azure.")
            litellm.api_version = api_version or os.getenv("AZURE_API_VERSION")
        else:
            raise ValueError(f"Provider is not recognized.")

    def _generate_batch_response(self, text, system_message, provider, model_config, api_key, api_base):
        """Generate a batch of responses using the specified provider."""
        MAX_RETRIES = 3
        
        for attempt in range(MAX_RETRIES):
            try:
                if provider == "gemini" and api_base:
                    messages = [{'role': 'user', 'content': system_message + text}]
                    response = proxy_api_completion(messages=messages, model=model_config["model"], api_base=api_base)
                    # response = proxy_call.api_completion(messages=messages, model=model_config["model"], api_base=api_base)
                    return pd.DataFrame(ast.literal_eval(response[0]))
                else:
                    return self._generate_llm_response(text, system_message, model_config, api_key)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Failed to generate valid response after {MAX_RETRIES} attempts: {str(e)}")
                continue

    def _generate_internal_response(self, text, system_message, model_config, kwargs):
        """Generate response using internal API."""
        messages = [{'role': 'user', 'content': system_message + text}]
        return internal_api_completion(
            messages=messages,
            model_config=model_config,
            kwargs=kwargs
        )

    def validate_input(self,text):

        if not text.strip():
            return 'Empty Text provided for qna generation. Please provide valid text'
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(text)
        if len(tokens)<5:
            return 'Very Small Text provided for qna generation. Please provide longer text'
        return False
        
          
    def _get_system_message(self, question_type, n):
        """
        Get the appropriate system message for the specified question type.

        Args:
            question_type (str): The type of questions to generate ('simple', 'mcq', or 'complex').
            n (int): The number of question/answer pairs to generate.

        Returns:
            str: The system message for the AI model.

        Raises:
            ValueError: If an invalid question type is specified.
        """
        if question_type == 'simple':
            return f'''Generate a set of {n} very simple questions answerable in a single phrase using the below text.
                Only generate questions answerable from the text given, to cover all parts of the given document. 
                Also return the answers for the generated questions.
                Return the response in a list of object format. 
                Each object in list should have Question and corresponding answer.
                Do not return any extra strings. Return Generated text strictly in below format.  
                [{{"Question":"question,"Answer":"answer"}}]
            '''
        elif question_type == 'mcq':
            return f'''Generate a set of {n} questions with 4 probable answers from the given text. 
                Only generate questions answerable from the text given, to cover all parts of the given document. 
                The options should not be longer than a phrase. There should be only 1 correct answer.
                There should not be any ambiguity between correct and incorrect options.
                Return the response in a list of object format. 
                Each object in list should have Question and a list of options. 
                Do not return any extra strings. Return Generated text strictly in below format. 
                [{{"Question":"question","Options":[option1,option2,option3,option4]}}]
            '''
        elif question_type == 'complex':
            return f'''Can you generate a set of {n} complex questions answerable in long form from the below texts.
                Only generate questions answerable from the text given, to cover all parts of the given document. 
                Make sure the questions are important and provide new information to the user.
                Return the response in a list of object format. Enclose any quotes in single quote. 
                Do not use double quotes within questions or answers.
                Each object in list should have Question and corresponding answer.
                Do not return any extra strings. Return generated text strictly in below format.
                [{{"Question":"question","Answer":"answers"}}]
            '''
        else:
            raise ValueError("Invalid question type")

    def _generate_llm_response(self, text, system_message, model_config, api_key=None):
        """
        Generate questions using LiteLLM which supports multiple providers (OpenAI, Groq, Gemini, etc.).

        Args:
            text (str): The input text to generate questions from.
            system_message (str): The system message for the AI model.
            model_config (dict): Configuration dictionary containing model details.
                Required keys:
                - model: The model identifier (e.g., "gpt-4", "gemini-pro", "mixtral-8x7b-32768")
                Optional keys:
                - api_base: Custom API base URL if needed
                - max_tokens: Maximum tokens in response
                - temperature: Temperature for response generation
            api_key (str, optional): The API key for the model provider.

        Returns:
            pandas.DataFrame: A DataFrame containing the generated questions and answers.

        Raises:
            Exception: If there's an error in generating the response.
        """

            # Prepare the messages in the format expected by LiteLLM
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]

        # Set up the completion parameters
        completion_params = {
            "model": model_config["model"],
            "messages": messages,
            "api_key": api_key
        }

        # Add optional parameters if they exist in model_config
        if "api_base" in model_config:
            completion_params["api_base"] = model_config["api_base"]
        if "api_version" in model_config:
            completion_params["api_version"] = model_config["api_version"]
        if "max_tokens" in model_config:
            completion_params["max_tokens"] = model_config["max_tokens"]
        if "temperature" in model_config:
            completion_params["temperature"] = model_config["temperature"]
        if 'provider' in model_config:
            completion_params['model'] = f'{model_config["provider"]}/{model_config["model"]}'

        # Make the API call using LiteLLM
        try:
            response = completion(**completion_params)
        except Exception as e:
            if any(error in str(e).lower() for error in ["invalid api key", "incorrect api key", "unauthorized", "authentication"]):
                raise ValueError(f"Invalid API key provided for {model_config.get('provider', 'the specified')} provider")
            raise Exception(f"Error calling LLM API: {str(e)}")

        # Extract the content from the response
        content = response.choices[0].message.content
        content = content.replace('\n', '').replace('```json','').replace('```', '').strip()

        # Clean the response if needed (remove any prefix before the JSON list)
        list_start_index = content.find('[')
        if list_start_index != -1:
            content = content[list_start_index:]

        json_data = json.loads(content)
        return pd.DataFrame(json_data)
    
    def _generate_raw_llm_response(self, text, system_message: Optional[str] = None, model_config: Dict[str, Any] = dict(), api_key=None):
        """
        Generate questions using LiteLLM which supports multiple providers (OpenAI, Groq, Gemini, etc.).

        Args:
            text (str): The input text to generate questions from.
            system_message (str): The system message for the AI model.
            model_config (dict): Configuration dictionary containing model details.
                Required keys:
                - model: The model identifier (e.g., "gpt-4", "gemini-pro", "mixtral-8x7b-32768")
                Optional keys:
                - api_base: Custom API base URL if needed
                - max_tokens: Maximum tokens in response
                - temperature: Temperature for response generation
            api_key (str, optional): The API key for the model provider.

        Returns:
            pandas.DataFrame: A DataFrame containing the generated questions and answers.

        Raises:
            Exception: If there's an error in generating the response.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]

        completion_params = {
            "model": model_config.get("model", 'gpt-4o'),
            "messages": messages,
            "api_key": api_key
        }

        if "api_base" in model_config:
            completion_params["api_base"] = model_config["api_base"]
        if "api_version" in model_config:
            completion_params["api_version"] = model_config["api_version"]
        if "max_tokens" in model_config:
            completion_params["max_tokens"] = model_config["max_tokens"]
        if "temperature" in model_config:
            completion_params["temperature"] = model_config["temperature"]
        if 'provider' in model_config:
            completion_params['model'] = f'{model_config["provider"]}/{model_config["model"]}'

        try:
            response = completion(**completion_params)
        except Exception as e:
            if any(error in str(e).lower() for error in ["invalid api key", "incorrect api key", "unauthorized", "authentication"]):
                raise ValueError(f"Invalid API key provided for {model_config.get('provider', 'the specified')} provider")
            raise Exception(f"Error calling LLM API: {str(e)}")

        return response.choices[0].message.content

    def _parse_response(self, response, provider):
        """
        Parse the response from the AI model and return it as a DataFrame.

        Args:
            response (str): The response from the AI model.
            provider (str): The AI provider used ('groq', 'gemini', or 'openai').
        Returns:
            pandas.DataFrame: The parsed response as a DataFrame.
        """
        if provider == "openai":
            data = response.choices[0].message.content
        elif provider == "gemini":
            data = response.candidates[0].content.parts[0].text
        elif provider == "groq":
            data = response.choices[0].message.content.replace('\n', '')
            list_start_index = data.find('[')  # Find the index of the first '['
            substring_data = data[list_start_index:] if list_start_index != -1 else data  # Slice from the list start
            data = substring_data
        elif provider == "azure":
            data = response.choices[0].message.content.replace('\n', '')
            list_start_index = data.find('[')  # Find the index of the first '['
            substring_data = data[list_start_index:] if list_start_index != -1 else data  # Slice from the list start
            data = substring_data
        else:
            raise ValueError("Invalid provider. Choose 'groq', 'gemini', 'azure' or 'openai'.")
        try:
            json_data = json.loads(data)
            return pd.DataFrame(json_data)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a DataFrame with a single column
            return pd.DataFrame({'content': [data]})

    def process_document(self, input_data):
        """
        Process the input document and extract its content.

        Args:
            input_data (str): Either a file path or a string of text.

        Returns:
            str: The extracted text content from the document.

        Raises:
            ValueError: If the input is neither a valid file path nor a string of text.
        """
        if isinstance(input_data, str):
            if os.path.isfile(input_data):
                # If input_data is a file path
                _, file_extension = os.path.splitext(input_data)
                try:
                    if file_extension.lower() == '.pdf':
                        return self._read_pdf(input_data)
                    elif file_extension.lower() == '.txt':
                        return self._read_text(input_data)
                    elif file_extension.lower() == '.md':
                        return self._read_markdown(input_data)
                    elif file_extension.lower() == '.csv':
                        return self._read_csv(input_data)
                    else:
                        raise ValueError(f"Unsupported file type: {file_extension}")
                except Exception as e:
                    raise ValueError(f"Error reading the file. Upload a valid file. \n{e}")
            else:
                # If input_data is a string of text
                return input_data
        else:
            raise ValueError("Input must be either a file path or a string of text")

    def _read_pdf(self, file_path):
        """
        Read and extract text from a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text content from the PDF.
        """
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def _read_text(self, file_path):
        """
        Read the contents of a text file.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The contents of the text file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _read_markdown(self, file_path):
        """
        Read and convert a Markdown file to HTML.

        Args:
            file_path (str): The path to the Markdown file.

        Returns:
            str: The HTML content converted from the Markdown file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
            html_content = markdown.markdown(md_content)
            return html_content

    def _read_csv(self, file_path):
        """
        Read and extract text from a CSV file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            str: The extracted text content from the CSV, with each row joined and separated by newlines.
        """
        text = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                text += " ".join(row) + "\n"
        return text

    def get_supported_qna(self):
        """
        Get a list of supported question types.

        Returns:
            list: A list of supported question types.
        """
        return ['simple', 'mcq', 'complex']

    def get_supported_providers(self):
        """
        Get a list of supported AI providers.

        Returns:
            list: A list of supported AI providers.
        """
        return ['gemini', 'openai','azure']
    
    def _get_init_ex_gen_prompt(self):
        prompt = '''
You are an expert example generator. Your task is to produce creative, relevant and varied examples according to the user instructions. 

**Inputs**
User Instruction: The user will provide guidance on how to generate examples, possibly accompanied by their own examples.
User Examples[Optional]: The user may supply examples.
User Context[Optional]: The user may supply context to generate the examples from.
No of Examples: The total number of examples to produce.

**Steps to follow**
1. Carefully analyze the user's instruction
2. If user examples are provided, check whether the user’s instructions refer to them specifically.
3. If user context is provided, understand it thoroughly and identify relevant parts to generate examples.
4. Comply with the system’s guidelines to generate examples, incorporating any user examples or user context as needed.

**Output Format**:  
- Present examples in a multiline string with each line a separate example.  
- Avoid markdown or special formatting.
- Omit any boilerplate texts.

**Instructions for Diversity**:  
- Vary the examples by context, tone, and (if applicable) technical complexity.
- Include edge cases or unconventional scenarios.  
- Ensure no two examples are conceptually identical.

**Final Notes**:  
- Focus on both originality and practical relevance.
- Avoid repetitiveness in the examples.
'''
        return prompt
    
    def _get_iter_ex_gen_prompt(self):
        prompt = '''
You are an expert example generator. Your task is to produce creative, relevant and varied examples according to the user instructions. 

**Inputs**
User Instruction: The user will provide guidance on how to generate examples, possibly accompanied by their own examples.
User Examples[Optional]: The user may supply examples.
User Context[Optional]: The user may supply context to generate the examples from.
No of Examples: The total number of examples to produce.
Relevant Examples: Any examples that are relevant to the user's instruction.
Irrelevant Examples: Any examples that are not relevant to the user's instruction.

**Steps to follow**
1. Carefully analyze the user's instruction
2. If user examples are provided, check whether the user’s instructions refer to them specifically.
3. If user context is provided, understand it thoroughly and identify relevant parts to generate examples.
4. Review the relevant and irrelevant examples present, understanding the differences in them.
5. Comply with the user's instruction to generate examples, similar to relevant examples and dissimilar to irrelevant ones.

**Output Format**:  
- Present examples in a multiline sting with each line a separate example.  
- Avoid markdown or special formatting.
- Omit any boilerplate texts.

**Instructions for Diversity**:  
- Vary the examples by context, tone, and (if applicable) technical complexity.
- Include edge cases or unconventional scenarios.  
- Ensure no two examples are conceptually identical.

**Final Notes**:  
- Focus on both originality and practical relevance.
- Avoid repetitiveness in the examples.
'''
        return prompt
    
    def _generate_examples_iter(
            self, 
            user_instruction: str, 
            user_examples: Optional[List[str] | str] = None, 
            user_context: Optional[str] = None, 
            relevant_examples: List[str]=[], 
            irrelevant_examples: List[str]=[], 
            no_examples: Optional[int] = None, 
            model_config: Dict[str, Any] = dict(), 
            api_key: Optional[str] = None
            ):
        if no_examples is None:
            no_examples = 5
        relevant_examples_str = '\n'.join(relevant_examples)
        irrelevant_examples_str = '\n'.join(irrelevant_examples)
        user_message = f'**User Instruction:** {user_instruction}'
        user_message += f'\n\n**No of Examples:** {no_examples}'
        if user_examples:
            if isinstance(user_examples, str):
                user_examples_str = user_examples
            elif isinstance(user_examples, list):
                user_examples_str = "\n".join(user_examples)
            else:
                raise ValueError(f'Expected string or list of strings as user_examples got {type(user_examples)}')
            user_message += f"\n\n**User Examples:** \n{user_examples_str}"
        if relevant_examples:
            user_message += f'\n\n**Relevant Examples:** \n{relevant_examples_str}'
        if irrelevant_examples:
            user_message += f'\n\n**Irrelevant Examples:** \n{irrelevant_examples_str}'
        if user_context:
            user_message += f'\n\n**User Context:** \n{user_context}'
        system_prompt = self._get_iter_ex_gen_prompt()
        return self._generate_raw_llm_response(user_message, system_prompt, model_config=model_config, api_key=api_key)
    
    def _generate_examples(
            self, 
            user_instruction:str, 
            user_examples:Optional[List[str]|str]=None, 
            user_context: Optional[str] = None, 
            no_examples:Optional[int]=None, 
            model_config: Dict[str, Any] = dict(), 
            api_key: Optional[str] = None
            ):
        if no_examples is None:
            no_examples = 5
        user_message = f"**User Instruction:** {user_instruction}"
        if user_examples:
            if isinstance(user_examples, str):
                user_examples_str = user_examples
            elif isinstance(user_examples, list):
                user_examples_str = "\n".join(user_examples)
            else:
                raise ValueError(f'Expected string or list of strings as user_examples got {type(user_examples)}')
            user_message += f"\n\n**User Examples:** \n{user_examples_str}"
        if user_context:
            user_message += f'\n\n**User Context:** \n{user_context}'
        user_message += f'\n\n**No of Examples:** {no_examples}'
        init_system_prompt = self._get_init_ex_gen_prompt()
        return self._generate_raw_llm_response(user_message, init_system_prompt, model_config=model_config, api_key=api_key)
    
    def _get_valid_examples(self, user_indices_str: str, examples: List[str]):
        valid_examples = []
        try:
            user_indices = user_indices_str.strip().split(',')
            for index_str in user_indices:
                try:
                    index = int(index_str)
                    if index <= 0 or index > len(examples):
                        continue
                except ValueError as e:
                    continue
                valid_examples.append(examples[index-1])
        except Exception as e:
            print(f'Error: {e}')
        return valid_examples
    
    def generate_examples(
        self, 
        user_instruction: str, 
        user_examples:Optional[List[str] | str] = None, 
        user_context: Optional[str] = None, 
        no_examples: Optional[int] = None, 
        model_config: Optional[Dict[str, Any]] = None, 
        api_key: Optional[str] = None, 
        max_iter: int = 0,
        **kwargs
        ):
        if not model_config:
            model_config = {}
        provider = model_config.get("provider")
        api_base = model_config.get("api_base")
        api_version = model_config.get("api_version")
        self._initialize_client(provider, api_key, api_base, api_version, internal_llm_proxy=kwargs.get("internal_llm_proxy", None))

        if no_examples is None:
            no_examples = 5
        assert no_examples >= 0, 'The number of examples cannot be less than 0'
        relevant_examples = []
        irrelevant_examples = []
        max_relevant_examples = 5
        max_irrelevant_examples = 10
        while len(relevant_examples) <= max_relevant_examples or len(irrelevant_examples) <= max_irrelevant_examples:
            if max_iter <= 0:
                break
            if len(relevant_examples) > max_relevant_examples:
                relevant_examples = random.sample(relevant_examples, max_relevant_examples)
            if len(irrelevant_examples) > max_irrelevant_examples:
                irrelevant_examples = random.sample(irrelevant_examples, max_irrelevant_examples)
            if relevant_examples or irrelevant_examples:
                examples_str = self._generate_examples_iter(
                    user_instruction = user_instruction, 
                    user_examples = user_examples, 
                    relevant_examples = relevant_examples, 
                    irrelevant_examples = irrelevant_examples, 
                    model_config = model_config, 
                    api_key = api_key
                    )
            else:
                examples_str = self._generate_examples(
                    user_instruction = user_instruction, 
                    user_examples = user_examples, 
                    user_context = user_context, 
                    model_config = model_config, 
                    api_key = api_key
                )
            examples = [example for example in examples_str.split('\n') if example.strip()]
            print('Generated Examples:')
            for i, example in enumerate(examples):
                print(f'{i+1}. {example}')
            relevant_indices = input('Enter the indices of relevant examples (comma-separated): ').strip()
            if relevant_indices:
                relevant_examples.extend(self._get_valid_examples(relevant_indices, examples))
            irrelevant_indices = input('Enter the indices of irrelevant examples (comma-separated): ').strip()
            if irrelevant_indices:
                irrelevant_examples.extend(self._get_valid_examples(irrelevant_indices, examples))
            max_iter -= 1
        if len(relevant_examples) > max_relevant_examples:
            fin_relevant_examples = random.sample(relevant_examples, max_relevant_examples)
        else:
            fin_relevant_examples = relevant_examples
        if len(irrelevant_examples) > max_irrelevant_examples:
            fin_irrelevant_examples = random.sample(irrelevant_examples, max_irrelevant_examples)
        else:
            fin_irrelevant_examples = irrelevant_examples
        if relevant_examples or irrelevant_examples:
            if len(relevant_examples) < no_examples:
                more_no_examples = no_examples - len(relevant_examples)
                final_examples_str = self._generate_examples_iter(
                    user_instruction = user_instruction, 
                    user_examples = user_examples, 
                    user_context = user_context, 
                    relevant_examples = fin_relevant_examples, 
                    irrelevant_examples = fin_irrelevant_examples, 
                    no_examples = more_no_examples, 
                    model_config = model_config, 
                    api_key = api_key
                    )
                final_examples = [example for example in final_examples_str.split('\n') if example.strip()]
                final_examples.extend(relevant_examples)
            else:
                final_examples = random.sample(relevant_examples, no_examples)
        else:
            final_examples_str = self._generate_examples(
                user_instruction = user_instruction, 
                user_examples = user_examples, 
                user_context = user_context, 
                no_examples = no_examples, 
                model_config = model_config, 
                api_key = api_key
            )
            final_examples = [example for example in final_examples_str.split('\n') if example.strip()]
        return final_examples

    
    def generate_examples_from_csv(
            self, 
            csv_path: str, 
            dst_csv_path: Optional[str] = None, 
            no_examples: Optional[int] = None, 
            model_config: Optional[Dict[str, Any]] = None, 
            api_key: Optional[str] = None, 
            **kwargs
            ):
        if no_examples is None:
            no_examples = 5
        assert no_examples >= 0, 'The number of examples cannot be less than  0'
        df = pd.read_csv(csv_path)
        assert 'user_instruction' in df.columns, 'The csv must have a column named user_instruction'
        fin_df_list = []
        for i, row in df.iterrows():
            user_instruction = row['user_instruction']
            user_examples = row.get('user_examples')
            user_context = row.get('user_context')
            row_dict = row.to_dict()
            try:
                examples = self.generate_examples(
                    user_instruction = user_instruction, 
                    user_examples = user_examples, 
                    user_context = user_context, 
                    no_examples = no_examples, 
                    model_config = model_config, 
                    api_key = api_key
                )
            except Exception as e:
                continue
            row_dict['generated_examples'] = examples
            fin_df_list.append(row_dict)
        fin_df = pd.DataFrame(fin_df_list)
        csv_file, csv_ext = os.path.splitext(csv_path)
        if not dst_csv_path:
            dst_csv_path = csv_file + '_with_examples' + csv_ext
        dst_dir = os.path.dirname(dst_csv_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
        fin_df.to_csv(dst_csv_path)
        logger.info(f'CSV with generated examples saved at {dst_csv_path}')


# Usage:
# from synthetic_data_generation import SyntheticDataGeneration
# synthetic_data_generation = SyntheticDataGeneration()
# text = synthetic_data_generation.process_document(input_data=text_file)
# result = synthetic_data_generation.generate_question(text)
# supported_question_types = synthetic_data_generation.get_supported_question_types()
# supported_providers = synthetic_data_generation.get_supported_providers()
