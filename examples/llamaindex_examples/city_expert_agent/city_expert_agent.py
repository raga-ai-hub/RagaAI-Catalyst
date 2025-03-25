from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager
from pathlib import Path
import requests
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_ACCESS_KEY'),
    secret_key=os.getenv('RAGAAI_SECRET_KEY'),
    base_url="https://llm-dev5.ragaai.ai/api"
)
project = catalyst.create_project(
    project_name="city_expert",
    usecase="Agentic Application"
)
tracer = Tracer(
    project_name='city_expert',
    dataset_name='city_dataset',
    tracer_type="Agentic",
)
init_tracing(catalyst=catalyst, tracer=tracer)

tracer.start()

wiki_titles = [
    "Toronto", "Seattle", "Chicago", "Boston", "Houston", "Tokyo", "Berlin",
    "Lisbon", "Paris", "London", "Atlanta", "Munich", "Shanghai", "Beijing",
    "Copenhagen", "Moscow", "Cairo", "Karachi",
]

load_dotenv()

Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

node_parser = SentenceSplitter()

# Build agents dictionary
agents = {}
query_engines = {}

# this is for the baseline
all_nodes = []

@tracer.trace_custom("fetch_wiki_data")
def fetch_wiki_data():
    for title in wiki_titles:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page["extract"]

        data_path = Path("data")
        if not data_path.exists():
            Path.mkdir(data_path)

        with open(data_path / f"{title}.txt", "w") as fp:
            fp.write(wiki_text)

fetch_wiki_data()

city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

@tracer.trace_custom("build_indexes_and_agents")
def build_indexes_and_agents():
    for idx, wiki_title in enumerate(wiki_titles):
        nodes = node_parser.get_nodes_from_documents(city_docs[wiki_title])
        all_nodes.extend(nodes)

        if not os.path.exists(f"./data/{wiki_title}"):
            # build vector index
            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(
                persist_dir=f"./data/{wiki_title}"
            )
        else:
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f"./data/{wiki_title}"),
            )

        # build summary index
        summary_index = SummaryIndex(nodes)
        # define query engines
        vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
        summary_query_engine = summary_index.as_query_engine(llm=Settings.llm)

        # define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        f" {wiki_title} (e.g. the history, arts and culture,"
                        " sports, demographics, or more)."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        "Useful for any requests that require a holistic summary"
                        f" of EVERYTHING about {wiki_title}. For questions about"
                        " more specific sections, please use the vector_tool."
                    ),
                ),
            ),
        ]

        # build agent
        function_llm = OpenAI(model="gpt-4o-mini")
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=True,
            system_prompt=f"""\
You are a specialized agent designed to answer queries about {wiki_title}.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
        )

        agents[wiki_title] = agent
        query_engines[wiki_title] = vector_index.as_query_engine(
            similarity_top_k=2
        )

build_indexes_and_agents()

# define tool for each document agent
all_tools = []
for wiki_title in wiki_titles:
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        f" this tool if you want to answer any questions about {wiki_title}.\n"
    )
    doc_tool = QueryEngineTool(
        query_engine=agents[wiki_title],
        metadata=ToolMetadata(
            name=f"tool_{wiki_title}",
            description=wiki_summary,
        ),
    )
    all_tools.append(doc_tool)

# define an "object" index and retriever over these tools
obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

top_agent = OpenAIAgent.from_tools(
    tool_retriever=obj_index.as_retriever(similarity_top_k=3),
    system_prompt=""" \
You are an agent designed to answer queries about a set of given cities.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True,
)

base_index = VectorStoreIndex(all_nodes)
base_query_engine = base_index.as_query_engine(similarity_top_k=4)

# should use Boston agent -> vector tool
response = top_agent.query("Tell me about the arts and culture in Boston")

tracer.stop()
tracer.get_upload_status()