
import os
import re
import pandas as pd
from datetime import datetime
from PyPDF2 import PdfReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer


catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
    base_url=os.getenv('CATALYST_BASE_URL')
)
# Initialize tracer
tracer = Tracer(
    project_name=os.getenv('PROJECT_NAME'),
    dataset_name=os.getenv('DATASET_NAME'),
    tracer_type="agentic/llamaindex",
)
load_dotenv()

init_tracing(catalyst=catalyst, tracer=tracer)

DATA_DIR = "legal_data"
CASES_DIR = os.path.join(DATA_DIR, "cases")
STATUTES_CSV = os.path.join(DATA_DIR, "statutes.csv")

def parse_pdf_metadata(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = reader.pages[0].extract_text()
        
    date_match = re.search(r"DECISION_DATE: (\d{4}-\d{2}-\d{2})", text)
    date_str = datetime.strptime(date_match.group(1), "%Y-%m-%d").strftime("%Y-%m-%d")
    metadata = {
        "jurisdiction": re.search(r"JURISDICTION: (.+)", text).group(1),
        "decision_date": date_str,
        "cites": re.findall(r"CITES: (.+)", text)[0].split(", "), 
        "full_text": text
    }
    return text, metadata

def load_legal_data():
    nodes = []
    statutes_df = pd.read_csv(STATUTES_CSV)
    
    for filename in os.listdir(CASES_DIR):
        if filename.endswith(".pdf"):
            text, metadata = parse_pdf_metadata(os.path.join(CASES_DIR, filename))
            node = TextNode(
                text=text,
                metadata={**metadata, "filename": filename},
                excluded_embed_metadata_keys=["decision_date"]
            )
            nodes.append(node)
    
    return nodes, statutes_df

def main():
    if not os.path.exists(DATA_DIR):
        print("Error: Legal data not found. First run:")
        print("python create_sample_data.py")
        return
    
    nodes, statutes_df = load_legal_data()
    index = VectorStoreIndex(nodes)
    
    query_engine = RetrieverQueryEngine(
        retriever=VectorIndexRetriever(
            index=index,
            similarity_top_k=3
        ),
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="full_text")],
    )
    
    response = query_engine.query(
        "California employment law cases about overtime since 2020"
    )
    
    print("\nRelevant Cases:")
    for node in response.source_nodes:
        print(f"\n- {node.metadata['filename']}")
        print(f"  Jurisdiction: {node.metadata['jurisdiction']}")
        print(f"  Date: {node.metadata['decision_date']}")
        print(f"  Excerpt: {node.text[:200]}...")

if __name__ == "__main__":
    with tracer:
        main()