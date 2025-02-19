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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatLiteLLM(model="gpt-4o-mini")

messages = [
    HumanMessage(
        content="Translate this sentence from English to German. I love you."
    )
]
with tracer:
    response = chat(messages)
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
with tracer:
    response = chat(messages)
    tracer.add_context(context)


with tracer:
  for question in questions:
    response = process_document(source_doc_path, None, question)
    tracer.add_context(context)
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

Evaluate metrics on the uploaded tracer dataset.

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

#### Appending Metrics for New Data

If you've added new rows to your dataset, you can calculate metrics just for the new data:

```python
evaluation.append_metrics(display_name="Faithfulness_v1")
```
























