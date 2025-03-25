# Langraph Project

## Overview

This project is designed to facilitate the creation and management of prompt templates using a state graph workflow. It leverages the RagaAI Catalyst and Langchain libraries to interact with language models.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file (see below).

## Usage

To run the project, execute the following command:
```bash
python langraph.py
```

## Environment Variables

Ensure you have a `.env` file with the following variables:
- `ACCESS_KEY`: Your RagaAI Catalyst access key.
- `SECRET_KEY`: Your RagaAI Catalyst secret key.
- `BASE_URL`: The base URL for the RagaAI API.

## Requirements

See `requirements.txt` for a list of dependencies.

## License

This project is licensed under the MIT License.