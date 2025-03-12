import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
from haystack.telemetry import tutorial_running
from haystack.utils import fetch_archive_from_http

# Enable telemetry
tutorial_running(4)


document_store = InMemoryDocumentStore()

# Create a Retriever using embeddings
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    scale_score=False,
)


# Download the data
doc_dir = "data/tutorial4"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/small_faq_covid.csv.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

df = pd.read_csv(f"{doc_dir}/small_faq_covid.csv")
# Minimal cleaning
df.fillna(value="", inplace=True)
df["question"] = df["question"].apply(lambda x: x.strip())

# Create embeddings for our questions from the FAQs
questions = list(df["question"].values)
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"question": "content"})

# Convert Dataframe to list of dicts and index them in our DocumentStore
docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

# Initialize pipeline
pipe = FAQPipeline(retriever=retriever)


if __name__ == "__main__":
    # Ask questions
    questions = [
        "How is COVID-19 transmitted?",
        "What are the symptoms of COVID-19?",
        "How can I protect myself against COVID-19?",
        "Should I wear a mask and gloves when I go outside?",
    ]

    for question in questions[:1]:
        print(f"\nQuestion: {question}")
        prediction = pipe.run(query=question, params={"Retriever": {"top_k": 2}})
        answers = prediction["answers"]
        for answer in answers:
            print("-" * 100)
            print(f"Answer: {answer.answer}")
            print(f"Score: {answer.score:.4f}")
        print("#" * 100)
        
