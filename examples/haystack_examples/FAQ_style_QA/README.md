# FAQ-Style Question Answering with Haystack

This example demonstrates how to build a FAQ-style Question Answering system using Haystack's FAQPipeline to answer questions about COVID-19.

## Features

- Uses Haystack's InMemoryDocumentStore for efficient storage
- Implements semantic search using sentence-transformers embeddings
- Utilizes FAQPipeline for question answering
- Includes pre-built COVID-19 FAQ dataset
- Supports GPU acceleration for embeddings

## Setup

1. Install dependencies:
   ```python
   pip install -r requirements.txt
   ```

2. Copy `sample.env` to `.env` and configure if needed (optional - this example uses only open-source models)

3. Run the example:
   ```python
   python FAQ_style_QA.py
   ```

## How it Works

1. **Document Store Setup**:
   - Creates an InMemoryDocumentStore for storing FAQ documents
   - Downloads and processes a COVID-19 FAQ dataset

2. **Embedding Generation**:
   - Uses sentence-transformers model for generating embeddings
   - Embeds questions from the FAQ dataset
   - Stores documents with embeddings in the document store

3. **Question Answering**:
   - Uses FAQPipeline for semantic matching
   - Retrieves top-k most similar questions and their answers
   - Scores answers based on semantic similarity

## Example Usage

The example includes sample questions about COVID-19:
- How is COVID-19 transmitted?
- What are the symptoms of COVID-19?
- How can I protect myself against COVID-19?
- Should I wear a mask and gloves when I go outside?

## Customization

- Modify the embedding model by changing the model name in EmbeddingRetriever
- Adjust top_k parameter in the pipeline to get more or fewer answers
- Use your own FAQ dataset by modifying the data loading section
- Toggle GPU usage with the use_gpu parameter in EmbeddingRetriever
