# Email Data Extraction with OpenAI Agents SDK

This example demonstrates how to use the OpenAI Agents SDK with RagaAI Catalyst to extract structured information from emails.

## Overview

The application uses OpenAI's Agents SDK to parse unstructured email text and extract key information such as:
- Email subject and sender details
- Main discussion points
- Meeting information (date, time, location)
- Action items and tasks with assignees
- Next steps

The extracted data is structured using Pydantic models for easy manipulation and validation.

## Requirements

- Python 3.8+
- OpenAI API key
- RagaAI Catalyst credentials

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Copy [sample.env](cci:7://file:///Users/ragaai_user/work/ragaai-catalyst/examples/openai_agents_sdk/sample.env:0:0-0:0) to [.env](cci:7://file:///Users/ragaai_user/work/ragaai-catalyst/examples/openai_agents_sdk/sample.env:0:0-0:0) and fill in your API keys:
```bash
cp sample.env .env
```

## Environment Variables

Configure the following environment variables in your [.env](cci:7://file:///Users/ragaai_user/work/ragaai-catalyst/examples/openai_agents_sdk/sample.env:0:0-0:0) file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `CATALYST_ACCESS_KEY`: Your RagaAI Catalyst access key
- `CATALYST_SECRET_KEY`: Your RagaAI Catalyst secret key
- `CATALYST_BASE_URL`: RagaAI Catalyst base URL
- `PROJECT_NAME`: Name for your project in RagaAI Catalyst (default: 'email-extraction')
- `DATASET_NAME`: Name for your dataset in RagaAI Catalyst (default: 'email-data')

## Usage

Run the example script:

```bash
python data_extraction_email.py
```
The script will:

1. Initialize the RagaAI Catalyst client for tracing
2. Set up an OpenAI Agent with appropriate instructions
3. Process a sample email to extract structured data
4. Display the extracted information

## Customization

You can modify the `sample_email` variable in the script to process different emails, or adapt the code to read emails from files or an API.

The Pydantic models (`Person`, `Meeting`, `Task`, `EmailData`) can be extended to capture additional information as needed.

## Integration with RagaAI Catalyst

This example integrates with RagaAI Catalyst for tracing and monitoring agent interactions. The integration helps with:

- Tracking agent performance
- Debugging complex agent workflows
- Collecting data for future improvements
