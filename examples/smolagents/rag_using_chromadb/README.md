# Hugging Face Documentation RAG using ChromaDB

This example demonstrates how to build a Retrieval Augmented Generation (RAG) system using SmoLAgents and LangChain to answer questions about Hugging Face documentation.

## Features

- Downloads and processes Hugging Face documentation from the `m-ric/huggingface_doc` dataset
- Uses LangChain for document processing and text splitting
- Implements semantic search using ChromaDB and sentence transformers
- Provides a custom RetrieverTool for SmoLAgents to perform documentation searches
- Supports multiple LLM backends (Groq, Anthropic, OpenAI, Hugging Face)

## Setup

1. Install dependencies:
   ```python
   pip install -r requirements.txt
   ```

2. Copy `sample.env` to `.env` and add your API key for your chosen LLM provider:
   - Groq API key for using `groq/llama-3.3-70b-versatile`
   - Anthropic API key for using `anthropic/claude-3-5-sonnet`
   - OpenAI API key for using OpenAI models
   - Hugging Face API key for using Hugging Face models

3. Run the example:
   ```python
   python rag_using_chromadb.py
   ```

## How it Works

1. **Document Processing**:
   - Loads Hugging Face documentation from the dataset
   - Splits documents into chunks using RecursiveCharacterTextSplitter
   - Removes duplicate content

2. **Vector Store**:
   - Uses sentence-transformers to create embeddings
   - Stores embeddings in ChromaDB for efficient retrieval

3. **RetrieverTool**:
   - Custom SmoLAgents tool for semantic search
   - Returns top 3 most relevant documentation snippets

4. **Agent Interaction**:
   - Uses CodeAgent to process queries
   - Retrieves relevant documentation
   - Generates responses based on retrieved content

## Example Usage

The example demonstrates answering the question "How can I push a model to the Hub?". You can modify the query in `rag_using_chromadb.py` to ask different questions about Hugging Face functionality.

## Customization

- Change the LLM model by uncommenting different model configurations
- Adjust document chunk size and overlap in the text splitter
- Modify the number of retrieved documents in RetrieverTool
- Use your own PDF documents by uncommenting the PDF loader code
