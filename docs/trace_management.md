## Trace Management

Record and analyse trace using the `ragaai_catalyst` library. This guide provides steps to initialize tracer with project and dataset name(langchain and llama-index),run tracer and add context,stop the tracer,list dataset,add rows and column and evalutaion on tracer datasets efficiently.

#### Initialize Tracer Management

To start managing datasets for a specific project, initialize the `Tracer` class with your project name.

##### 1. langchain example

```python
from ragaai_catalyst import Tracer
tracer_dataset_name = "tracer_dataset_name"


tracer = Tracer(
    project_name=project_name,
    dataset_name=tracer_dataset_name,
    metadata={"key1": "value1", "key2": "value2"},
    tracer_type="langchain",
    pipeline={
        "llm_model": "gpt-4o-mini",
        "vector_store": "faiss",
        "embed_model": "text-embedding-ada-002",
    }
)
```
##### - User code

```python
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_radiology_prompt(resp_dict: dict, format_instructions: str) -> ChatPromptTemplate:
    enhanced_instructions = f"""Output must strictly adhere to this format: {format_instructions}. Do not add any explanations or extra text"""
    impression_system = """You are an expert radiologist responsible for generating precise and focused impressions for radiology reports. Your impressions should adhere to current clinical guidelines and best practices from reputable radiological societies such as the ACR (American College of Radiology), RSNA (Radiological Society of North America), Fleischner Society, and other relevant organizations. Incorporate recommendations and guidelines from clinical papers and established frameworks to ensure optimal patient care.
    {enhanced_instructions}"""
    impression_prompt = """
    You are a knowledgeable radiologist responsible for crafting precise and focused impressions for radiology reports. Your impressions should be clear, concise, and adhere to the following guidelines.
        PRIORITY ORDER:
        Clinical Findings:
        Start with findings directly related to the reason for the exam or the primary clinical indications.

        Acute Findings Statement:
        Clearly state the presence or absence of acute findings in the examined regions.

        Post-Surgical Changes:
        Note any stable post-surgical changes relevant to the patient's history.

        Clinically Significant Findings:
        List clinically significant findings in order of importance, providing essential details.

        Unremarkable Exam Statement:
        If the entire exam findings are unremarkable and there are no clinical findings, simply state "Unremarkable exam." and do not include any other statements in the impression.

        CORE RULES:
        Exclude Normal and Negative Findings:
        Do not include normal findings or negative statements about structures without abnormalities unless directly addressing the clinical question.
        Avoid phrases like "No evidence of..." or "No significant abnormalities in...".

        Include Only Essential Findings:
        Focus on clinically significant findings that impact patient care or require follow-up.

        Include Recommendations When Necessary:
        Provide recommendations if they significantly impact patient management or are essential for follow-up.

        Avoid Redundant Phrases:
        Do not use phrases like "present," "noted," "identified," or "visualized."

        Exclude Unnecessary Details:
        Do not include technical details, unnecessary measurements, or incidental findings without clinical relevance.

        STYLE GUIDELINES:
        Numbered List:
        Always present the Impression as a numbered list, with each point numbered and on its own line for clarity.

        Conciseness:
        Keep the impression concise, using only as many lines as necessary for essential information.

        Order of Significance:
        List findings from most significant to least significant.

        Consistent Wording:
        Use clear and consistent terminology, avoiding unnecessary words or phrases.

        Punctuation:
        Use periods at the end of each statement.

        FORMAT SUMMARY:
        Clinical Findings:
        Address findings related to the clinical indications for the exam.

        Acute Findings Statement:
        State the presence or absence of acute findings relevant to the clinical concern.

        Post-Surgical Changes:
        Mention stable post-surgical changes if applicable.

        Clinically Significant Findings:
        Include significant findings with essential details, ordered by importance.

        Unremarkable Exam Statement:
        If the exam is completely unremarkable and there are no clinical findings, state "Unremarkable exam." and do not include any other statements in the impression.

        Your Task:
        Generate an IMPRESSION section based on the provided findings, strictly following these guidelines. Focus on conveying critical information to the referring clinician, ensuring the impression is concise and aligns with the known good impression in focus and content.

        Note:
        Only output the Impression text, presented as a numbered list.
        Do not include normal findings or unnecessary negative statements unless directly relevant to the clinical indication.
        Include significant recommendations if they impact patient management.
        If the exam is completely unremarkable and there are no clinical findings, state "Unremarkable exam." and do not include any other statements in the impression.
        Patient-Information:
        Age: {age}
        Gender: {gender}
        Ethnicity: {ethnicity}

        Radiography Exam:
        Type of Radiography Exam: {exam}
        Reason For Exam: {reason_for_exam}
        Technique: {technique}
        Indication Codes List: {indication_codes}

        Findings: {findings}

        """

    context = {
        "age": resp_dict.get("age"),
        "gender": resp_dict.get("gender"), "ethnicity": resp_dict.get("ethnicity"),
        "exam": resp_dict.get("exam"), "reason_for_exam": resp_dict.get("reason_for_exam"),
        "technique": resp_dict.get("technique"), "indication_codes": resp_dict.get("indication_codes"),
        "findings": resp_dict.get("findings"), "format_instructions": format_instructions
    }
    system_message = SystemMessagePromptTemplate.from_template(impression_system, partial_variables={"enhanced_instructions": enhanced_instructions})

    human_message = HumanMessagePromptTemplate.from_template(impression_prompt, partial_variables=context)

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    return prompt, context
```

```python
!pip install langchain_community -q
```

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI, VertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)

class ImpressionPromptOutput(BaseModel):
    impression: str = Field(description="Impression generated by LLM")

# Define model configurations
# Define model configurations
MODEL_CONFIGS = {
    # Anthropic Models
    "claude-3-5-sonnet-latest": {
        "provider": "anthropic",
        "model_class": ChatAnthropic,
        "kwargs": {
            "model": "claude-3-5-sonnet-latest",
            "temperature": 0.7
        }
    },
    # OpenAI Models
    "gpt-4o": {
        "provider": "openai",
        "model_class": ChatOpenAI,
        "kwargs": {
            "model_name": "gpt-4o",
            "temperature": 0.7
        }
    }
}
```

```python
def generate(resp_dict, model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    model = config["model_class"](**config["kwargs"])
    parser = PydanticOutputParser(pydantic_object=ImpressionPromptOutput)
    prompt, context = get_radiology_prompt(resp_dict, parser.get_format_instructions())

    try:
        chain = prompt | model | parser
        output = chain.invoke({})
        print(f"Raw LLM Response: {output}")
    except Exception as e:
        logger.warning(f"Errors during chain execution: {e}")
        try:
            chain = prompt | model
            output = chain.invoke({})
            print(f"Raw LLM Response: here::::{output}")
            try:
                wrapped_resp = json.dumps({"impression": output})
            except:
                wrapped_resp = json.dumps({"impression": output.content})
            output = parser.parse(wrapped_resp)
        except Exception as fallback_error:
            logger.warning(f"Errors during chain fallback execution: {fallback_error}")
            output = None

    return output, context

def get_impression(resp_dict, model_name="gemini-1.5-flash-002"):
    new_impression, context = generate(resp_dict, model_name)
    return new_impression, context
```

##### 2. Llama-index example

```python
from ragaai_catalyst import Tracer
tracer_dataset_name = "tracer_dataset_name"

tracer = Tracer(
    project_name=project_name,
    dataset_name=tracer_dataset_name,
    metadata={"key1": "value1", "key2": "value2"},
    tracer_type="llamaindex",
    pipeline={
        "llm_model": "gpt-4o-mini",
        "vector_store": "faiss",
        "embed_model": "text-embedding-ada-002",
    }
)
```
##### - User code

```python
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize necessary variables
retriever = None
loaded_doc = None
index = None

def load_document(source_doc_path):
    """
    Load and index the document using LlamaIndex
    """
    try:
        # Initialize LLM and embedding model
        Settings.llm = OpenAI(model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding()


        # Load PDF document
        reader = PDFReader()
        docs = reader.load_data(source_doc_path)

        # Create documents with metadata
        documents = [
            Document(text=doc.text, metadata={"source": source_doc_path})
            for doc in docs
        ]

        # Create vector store index
        global index
        index = VectorStoreIndex.from_documents(documents)

        # Create retriever (to maintain similar interface)
        retriever = index.as_retriever(similarity_top_k=5)

        logger.info("Document loaded and processed.")
        return retriever

    except Exception as e:
        logger.error(f"An error occurred while loading the document: {e}")
        return None

def generate_response(retriever, query):
    """
    Generate response for the given query using LlamaIndex
    """
    try:
        if index is None:
            logger.error("Index not initialized. Please load document first.")
            return None

        # Create query engine
        query_engine = index.as_query_engine(
            response_mode="compact"
        )

        # Generate response
        response = query_engine.query(query)

        logger.info("Response generated successfully")
        return str(response)

    except Exception as e:
        logger.error(f"An error occurred while generating the response: {e}")
        return None

def process_document(source_doc_path, loaded_doc, query):
    """
    Process document and generate response using LlamaIndex
    """
    try:
        # Check if we need to load a new document
        if loaded_doc != source_doc_path:
            retriever = load_document(source_doc_path)
            if retriever is None:
                return "Failed to load document."
            loaded_doc = source_doc_path
        else:
            logger.info("Using cached document retriever.")

        # Generate response
        response = generate_response(retriever, query)
        if response is None:
            return "Failed to generate response."

        return response

    except Exception as e:
        logger.error(f"An overall error occurred: {e}")
        return "An error occurred during the document processing."



source_doc_path = "/content/2404.02798v1.pdf"

questions = [
    "What is this paper about?",
    "Give 10 words summary of the paper?",
    "What is the main topic of the paper?",
    "What is the aim of the paper, in 10 words?"
]
```

```python
with tracer:
  for question in questions:
    response = process_document(source_doc_path, None, question)
    print(f"Question: {question}\nResponse: {response}\n")
```

#### Run tracer and add context

You can add context using tracer.add_context(context).Context needs to be in str type

```python
# OpenAI
with tracer:
    result, context = get_impression(resp_dict, "gpt-4o")
    tracer.add_context(str(context))

# Anthropic
with tracer:
    result, context = get_impression(resp_dict, "claude-3-5-sonnet-latest")
    tracer.add_context(str(context))
```

#### Add rows to the uploaded tracer dataset

```python
from ragaai_catalyst import Dataset
dataset_manager = Dataset(project_name=project_name)
add_rows_csv_path = "path to dataset"
dataset_manager.add_rows(csv_path=add_rows_csv_path, dataset_name=dataset_name)
```

#### Add column to the uploaded tracer dataset

```python
text_fields = [
      {
        "role": "system",
        "content": "you are an evaluator, which answers only in yes or no."
      },
      {
        "role": "user",
        "content": "are any of the {{asdf}} {{abcd}} related to broken hand"
      }
    ]
column_name = "from_colab_v1"
provider = "openai"
model = "gpt-4o-mini"

variables={
    "asdf": "context",
    "abcd": "feedback"
}
```

```python
dataset_manager.add_columns(
    text_fields=text_fields,
    dataset_name=dataset_name,
    column_name=column_name,
    provider=provider,
    model=model,
    variables=variables
)
```

#### Evaluate metrics

Evaluate metrics on the uploaded dataset.

```python
from ragaai_catalyst import Evaluation
evaluation = Evaluation(project_name=project_name,
                        dataset_name=tracer_dataset_name)
```

```python                 
schema_mapping={
    'prompt': 'prompt',
    'response': 'response',
    'context': 'context',
}
metrics = [
     {"name": "Faithfulness", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.323}}, "column_name": "Faithfulness_v1_gte", "schema_mapping": schema_mapping},
     {"name": "Hallucination", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"lte": 0.323}}, "column_name": "Hallucination_v1_lte", "schema_mapping": schema_mapping},
     {"name": "Hallucination", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"eq": 0.323}}, "column_name": "Hallucination_v1_eq", "schema_mapping": schema_mapping},
     {"name": "Context Relevancy", "config": {"model": "gemini-1.5-flash", "provider": "gemini", "threshold": {"eq": 0.323}}, "column_name": "Context_Relevancy_v1_eq", "schema_mapping": schema_mapping},
    ]
```

```python
evaluation.add_metrics(metrics=metrics)
evaluation.get_status()
```
















