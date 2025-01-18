import os
from groq import Groq
import google.generativeai as genai
import openai
import PyPDF2
import csv
import markdown
import pandas as pd
import json
from litellm import completion
from tqdm import tqdm
# import internal_api_completion
# import proxy_call
from .internal_api_completion import api_completion as internal_api_completion
from .proxy_call import api_completion as proxy_api_completion
# from ragaai_catalyst import internal_api_completion
# from ragaai_catalyst import proxy_call
import ast

# dotenv.load_dotenv()

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
        BATCH_SIZE = 5  # Optimal batch size for maintaining response quality
        provider = model_config.get("provider")
        model = model_config.get("model")
        api_base = model_config.get("api_base")

        # Initialize the appropriate client based on provider
        self._initialize_client(provider, api_key, api_base, internal_llm_proxy=kwargs.get("internal_llm_proxy", None))

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
            "litellm.AuthenticationError"]
        
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
                print(f"Batch generation failed.")

                if any(error in str(e) for error in FAILURE_CASES):
                    raise Exception(f"{e}")

                else:
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

    def _initialize_client(self, provider, api_key, api_base=None, internal_llm_proxy=None):
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
            genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        
        elif provider == "openai":
            if api_key is None and os.getenv("OPENAI_API_KEY") is None and internal_llm_proxy is None:
                raise ValueError("API key must be provided for OpenAI.")
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
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
            return f'''Generate a set of {n} very simple questions answerable in a single phrase. 
                Also return the answers for the generated questions.
                Return the response in a list of object format. 
                Each object in list should have Question and corresponding answer.
                Do not return any extra strings. Return Generated text strictly in below format.  
                [{{"Question":"question,"Answer":"answer"}}]
            '''
        elif question_type == 'mcq':
            return f'''Generate a set of {n} questions with 4 probable answers from the given text. 
                The options should not be longer than a phrase. There should be only 1 correct answer.
                There should not be any ambiguity between correct and incorrect options.
                Return the response in a list of object format. 
                Each object in list should have Question and a list of options. 
                Do not return any extra strings. Return Generated text strictly in below format. 
                [{{"Question":"question","Options":[option1,option2,option3,option4]}}]
            '''
        elif question_type == 'complex':
            return f'''Can you generate a set of {n} complex questions answerable in long form from the below texts.
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
        if "max_tokens" in model_config:
            completion_params["max_tokens"] = model_config["max_tokens"]
        if "temperature" in model_config:
            completion_params["temperature"] = model_config["temperature"]

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

        else:
            raise ValueError("Invalid provider. Choose 'groq', 'gemini', or 'openai'.")
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
            pdf_reader = PyPDF2.PdfReader(file)
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
        return ['gemini', 'openai']

# Usage:
# from synthetic_data_generation import SyntheticDataGeneration
# synthetic_data_generation = SyntheticDataGeneration()
# text = synthetic_data_generation.process_document(input_data=text_file)
# result = synthetic_data_generation.generate_question(text)
# supported_question_types = synthetic_data_generation.get_supported_question_types()
# supported_providers = synthetic_data_generation.get_supported_providers()
