import os
import sys
import time
from pinecone import Pinecone, ServerlessSpec
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentneo import AgentNeo, Tracer, Evaluation,launch_dashboard
# Initialize AgentNeo session
# Create project
neo_session = AgentNeo(session_name="testingpine5")

project_name = "TestingPinconeV1"

try:
    neo_session.create_project(project_name=project_name)
    print("Project created successfully")
except:
    neo_session.connect_project(project_name=project_name)
    print("Project connected successfully")
# Start tracing
tracer = Tracer(session=neo_session)
tracer.start()

class MyPineconeApp():
    def __init__(self):
        super().__init__()
        api_key = os.getenv("PINECONE_API_KEY") or "YOUR-API-KEY"
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "pineconetesting172"
        self.dimension = 1024
        self.metric = "cosine"
        self.namespace = "ns1"
        self._initialize_pinecone_index()

    def _initialize_pinecone_index(self):
        existing_indexes = self.pc.list_indexes()
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        while True:
            index_status = self.pc.describe_index(self.index_name).status
            if index_status['ready']:
                break
            time.sleep(1)
        self.index = self.pc.Index(self.index_name)

    @tracer.trace_pinecone_upsert
    def upsert_data(self, index_name, data, namespace="ns1"):
        embeddings = self.pc.inference.embed(
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
        self.index.upsert(
            vectors=vectors,
            namespace=namespace
        )

    @tracer.trace_pinecone_create_query_vector
    def create_query_vector(self, query_text):
        embedding = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query_text],
            parameters={"input_type": "query"}
        )
        return embedding[0]['values']

    @tracer.trace_pinecone_similarity_search
    def similarity_search(self, index_name, query_vector, top_k=2, namespace="ns1"):
        results = self.index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k,
            score_threshold=-0.2,
            include_values=False,
            include_metadata=True
        )
        return results

    @tracer.trace_agent(name="PineconeVectorDbAgent")
    def tracing_agent(self):
        data = [
            {"id": "vec1", "text": "Apple is a leading tech company known for its innovative products like the iPhone, MacBook, and iPad, which have revolutionized the tech industry."},
            {"id": "vec2", "text": "Apple is a leading tech company known for its innovative products, including the iPhone, Apple Watch, and Mac computers, which have set industry standards."},
            {"id": "vec3", "text": "Apple is a leading tech company known for its innovative products that have reshaped personal computing, from the iPhone to the MacBook Pro."},
            {"id": "vec4", "text": "Apple is a leading tech company known for its innovative products such as the iPhone, MacBook, and Apple TV, which have transformed the digital lifestyle."},
            {"id": "vec5", "text": "Apple is a leading tech company known for its innovative products, particularly the iPhone and iPad, which have become icons in the tech world."},
            {"id": "vec6", "text": "Apple is a leading tech company known for its innovative products like the iPhone, iPad, and MacBook, and continues to push boundaries with new technologies."},
            {"id": "vec7", "text": "Apple is a leading tech company known for its innovative products that blend sleek design with cutting-edge technology, such as the iPhone, iPad, and MacBook."},
            {"id": "vec8", "text": "Apple is a leading tech company known for its innovative products, including the iPhone, iPad, and MacBook, which have set new standards in technology and user experience."},
            {"id": "vec9", "text": "Apple is a leading tech company known for its innovative products like the iPhone and iPad, which have redefined how we interact with technology."},
            {"id": "vec10", "text": "Apple is a leading tech company known for its innovative products, including the iPhone and MacBook, which have not only revolutionized the tech industry but also created a global ecosystem."},
            {"id": "vec11", "text": "Apple is a leading tech company known for its innovative products like the iPhone, MacBook, and AirPods, which have become synonymous with high quality and seamless integration."},
            {"id": "vec12", "text": "Apple is a leading tech company known for its innovative products such as the iPhone, iPad, and MacBook, which continue to inspire and shape the future of technology."},
            {"id": "vec13", "text": "Apple is a leading tech company known for its innovative products that combine form and function, such as the iPhone, MacBook, and the Apple ecosystem."},
            {"id": "vec14", "text": "Apple is a leading tech company known for its innovative products, from the iPhone to the Apple Watch, that have consistently pushed the limits of technology and design."},
            {"id": "vec15", "text": "Apple is a leading tech company known for its innovative products, including the iPhone, MacBook, and Apple Watch, which have revolutionized personal technology."}
        ]
        self.upsert_data(index_name="pineconetesting172", data=data)
        query_vector = self.create_query_vector(query_text="Apple is a leading tech company known for its innovative products")
        results = self.similarity_search(index_name="pineconetesting172", query_vector=query_vector)
        return results

# Usage Example
if __name__ == "__main__":
    app = MyPineconeApp()
    print("Strarting Pinecone Test")
    results=app.tracing_agent()
    print(results)
    tracer.stop()
    neo_session.launch_dashboard()