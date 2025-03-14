import os
import urllib.request
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder

from ragaai_catalyst import RagaAICatalyst, Tracer, init_tracing

from dotenv import load_dotenv
load_dotenv()

catalyst = RagaAICatalyst(
    access_key=os.environ['CATALYST_ACCESS_KEY'], 
    secret_key=os.environ['CATALYST_SECRET_KEY'], 
    base_url=os.environ['CATALYST_BASE_URL']
)

tracer = Tracer(
    project_name=os.environ['PROJECT_NAME'],
    dataset_name=os.environ['DATASET_NAME'],
    tracer_type="agentic/haystack",
)

init_tracing(catalyst=catalyst, tracer=tracer)

urllib.request.urlretrieve("https://archive.org/stream/leonardodavinci00brocrich/leonardodavinci00brocrich_djvu.txt",
                           "davinci.txt")    

document_store = InMemoryDocumentStore()

text_file_converter = TextFileToDocument()
cleaner = DocumentCleaner()
splitter = DocumentSplitter()
embedder = OpenAIDocumentEmbedder()
writer = DocumentWriter(document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", text_file_converter)
indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", writer)

indexing_pipeline.connect("converter.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")
indexing_pipeline.run(data={"sources": ["davinci.txt"]})

text_embedder = OpenAITextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store)
prompt_template = [
    ChatMessage.from_user(
      """
      Given these documents, answer the question.
      Documents:
      {% for doc in documents %}
          {{ doc.content }}
      {% endfor %}
      Question: {{query}}
      Answer:
      """
    )
]
prompt_builder = ChatPromptBuilder(template=prompt_template)
llm = OpenAIChatGenerator()

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")



query = "How old was Leonardo when he died?"
result = rag_pipeline.run(data={"prompt_builder": {"query":query}, "text_embedder": {"text": query}})


print(result["llm"]["replies"][0].text)