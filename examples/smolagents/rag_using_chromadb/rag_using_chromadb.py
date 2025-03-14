import os

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import LiteLLMModel, Tool
from smolagents.agents import CodeAgent
from tqdm import tqdm
from transformers import AutoTokenizer

from ragaai_catalyst import RagaAICatalyst, Tracer, init_tracing

from dotenv import load_dotenv
load_dotenv()

catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'),
    secret_key=os.getenv('CATALYST_SECRET_KEY'),
    base_url=os.getenv('CATALYST_BASE_URL'),
)

tracer = Tracer(
    project_name=os.getenv('PROJECT_NAME'),
    dataset_name=os.getenv('DATASET_NAME'),
    tracer_type="agentic/smolagents",
)

init_tracing(catalyst=catalyst, tracer=tracer)

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)


print(
    "Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)"
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vector_store = Chroma.from_documents(
    docs_processed, embeddings, persist_directory="./chroma_db"
)


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.vector_store.similarity_search(query, k=3)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


retriever_tool = RetrieverTool(vector_store)
model = LiteLLMModel(
    model_id="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
)

agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
)

if __name__ == "__main__":
    query = "How can I push a model to the Hub?"
    agent_output = agent.run(query)

    print("Final output:")
    print(agent_output)
