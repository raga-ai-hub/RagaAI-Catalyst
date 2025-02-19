import os
import sys
sys.path.append('/Users/ragaai_user/work/ragaai-catalyst/')
import time
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import dotenv
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from ragaai_catalyst import (
    Tracer, 
    RagaAICatalyst, 
    Evaluation
)
import pytest


catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAICATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAICATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAICATALYST_BASE_URL")
)

def create_rag_pipeline(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(texts, embeddings)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
    )
    
    return qa_chain

def run_pipeline():
    global tracer
    tracer = Tracer(
        project_name="langchain_tracing_test",
        dataset_name="test_00",
        tracer_type='langchain',
        metadata={
            "model": "gpt-3.5-turbo",
            "environment": "production"
        },
        pipeline={
            "llm_model": "gpt-3.5-turbo",
            "vector_store": "faiss",
            "embed_model": "text-embedding-ada-002",
        }
    )

    tracer.start()

    pdf_path = "test/test_catalyst/ai document_061023_2.pdf"

    qa_chain = create_rag_pipeline(pdf_path)
    
    questions = [
        "Provide specific title in 10 words about the doc"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = qa_chain.invoke(question)
        print(f"Answer: {result['result']}")
    tracer.stop()


class TestLangchainTracing:
    @classmethod
    def setup_class(cls):
        if os.path.exists('final_result.json'):
            os.remove('final_result.json')
        run_pipeline()
    
    @classmethod
    def teardown_class(cls):
        if os.path.exists('final_result.json'):
            os.remove('final_result.json')
    
    def test_final_result(self):
        assert os.path.exists('final_result.json'), "Final result file not created"
    
    def test_final_result_content(self):
        with open('final_result.json', 'r') as f:
            final_result = json.load(f)
        assert len(final_result) > 0, "Final result is empty"
    
    def test_traces_presence(self):
        with open('final_result.json', 'r') as f:
            final_result = json.load(f)
        assert 'traces' in final_result[0], "traces key not found in final result"
        traces = final_result[0]['traces']
        assert len(traces) > 0, "No traces found in final result"
    

    @pytest.mark.parametrize('part_name', [
        "retrieve_documents.langchain.workflow",
        "PromptTemplate.langchain.task", 
        "ChatOpenAI.langchain.task"
        ])
    def test_trace_parts(self, part_name):
        with open('final_result.json', 'r') as f:
            final_result = json.load(f)
        traces = final_result[0]['traces']
        parts = [trace['name'] for trace in traces]
        assert part_name in parts, f"{part_name} not found in final result"

    @pytest.mark.parametrize(('part_name', 'attr_len'), [
        ("retrieve_documents.langchain.workflow", 1),
        ("PromptTemplate.langchain.task", 2), 
        ("ChatOpenAI.langchain.task", 2)
        ])
    def test_traces(self, part_name, attr_len):
        with open('final_result.json', 'r') as f:
            final_result = json.load(f)
        traces = final_result[0]['traces']
        for trace in traces:
            if trace['name'] == part_name:
                assert len(trace['attributes']) == attr_len, f"{part_name} has incorrect number of attributes"

    def test_trace_metrics(self):
        with open('final_result.json', 'r') as f:
            final_result = json.load(f)
        trace_id = final_result[0]['trace_id']
        evaluation = Evaluation(
            project_name="langchain_tracing_test",
            dataset_name="test_00",
        )
        schema_mapping = {
            'prompt': 'prompt', 
            'response': 'response', 
            'context': 'context',
        }
        try:
            evaluation.add_metrics(
                metrics=[
                    {'name': 'Hallucination', 'config': {'model': 'gpt-4o-mini', 'provider': 'openai', 'threshold': {'gte': 0.3}}, 'column_name': 'Hallucination_v1', 'schema_mapping': schema_mapping},
                ]
            )
        except ValueError:
            evaluation.append_metrics('Hallucination_v1')
        except Exception:
            raise
        status = evaluation.get_status()
        while status == 'in_progress':
            time.sleep(3)
            status = evaluation.get_status()
        results = evaluation.get_results()
        assert len(results) > 0, "No results found"
        relevant_result  = results[results['trace_id'] == trace_id]
        assert len(relevant_result) > 0, "No relevant result found"
        assert not pd.isna(relevant_result['Hallucination_v1'].values[0]), "Hallucination metric is NaN"
