import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from agentneo import AgentNeo, Tracer
from pinecone import Pinecone, ServerlessSpec
from agentneo.data import ToolCallModel
import time

class PineconeTracing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize AgentNeo session
        cls.neo_session = AgentNeo(session_name="test_session")
        try:
            cls.neo_session.create_project(project_name="test_project")
            print("Project created successfully")
        except:
            cls.neo_session.connect_project(project_name="test_project")
            print("Project connected successfully")

        # Start tracing
        cls.tracer = Tracer(session=cls.neo_session)
        cls.tracer.start()

        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY") or "YOUR-API-KEY"
        cls.pc = Pinecone(api_key=api_key)
        cls.index_name = "pineconetestingunit9"
        cls.dimension = 1024
        cls.metric = "cosine"
        cls.namespace = "ns1"
        cls._initialize_pinecone_index()

    @classmethod
    def _initialize_pinecone_index(cls):
        existing_indexes = cls.pc.list_indexes()
        if cls.index_name not in existing_indexes:
            cls.pc.create_index(
                name=cls.index_name,
                dimension=cls.dimension,
                metric=cls.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        while True:
            index_status = cls.pc.describe_index(cls.index_name).status
            if index_status['ready']:
                break
            time.sleep(1)
        cls.index = cls.pc.Index(cls.index_name)

    def test_pinecone_tracing(self):
        # Prepare data to upsert
        data = [
            {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
            {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
        ]

        # Upsert data into Pinecone index and trace the operation
        self.upsert_data(index_name=self.index_name, data=data)

        # Create a query vector and trace the operation
        query_text = "Tell me about the tech company known as Apple."
        query_vector = self.create_query_vector(query_text=query_text)

        # Perform a similarity search and trace the operation
        self.similarity_search(index_name=self.index_name, query_vector=query_vector)

        # Stop tracing
        self.tracer.stop()

        # Verify that the Pinecone operations were traced
        with self.neo_session.Session() as session:
            tool_calls = session.query(ToolCallModel).filter_by(trace_id=self.tracer.trace_id).all()
            self.assertGreater(len(tool_calls), 0)
            self.assertEqual(tool_calls[0].name, "Pinecone Upsert")
            self.assertEqual(tool_calls[1].name, "Pinecone Create Query Vector")
            self.assertEqual(tool_calls[2].name, "Pinecone Similarity Search")

    @classmethod
    def upsert_data(cls, index_name, data, namespace="ns1"):
        @cls.tracer.trace_pinecone_upsert
        def _upsert_data():
            embeddings = cls.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[d['text'] for d in data],
                parameters={"input_type": "passage", "truncate": "END"}
            )
            vectors = []
            for d, e in zip(data, embeddings):
                vectors.append({
                    "id": d['id'],
                    "values": e['values'],
                    "metadata": {'text': d['text']}
                })
            cls.index.upsert(
                vectors=vectors,
                namespace=namespace
            )
        _upsert_data()

    @classmethod
    def create_query_vector(cls, query_text):
        @cls.tracer.trace_pinecone_create_query_vector
        def _create_query_vector():
            embedding = cls.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[query_text],
                parameters={"input_type": "query"}
            )
            return embedding[0]['values']
        return _create_query_vector()

    @classmethod
    def similarity_search(cls, index_name, query_vector, top_k=3, namespace="ns1"):
        @cls.tracer.trace_pinecone_similarity_search
        def _similarity_search():
            results = cls.index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )
            return results
        return _similarity_search()

if __name__ == '__main__':
    unittest.main()