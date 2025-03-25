# City Expert Agent

This project uses the Llama Index and RagaAI Catalyst to create specialized agents that can answer queries about various cities using Wikipedia data. The agents are designed to provide detailed information on specific aspects of each city, such as history, arts, culture, sports, and demographics.

## Features

- Fetches Wikipedia data for a list of cities.
- Builds vector and summary indexes for each city.
- Creates specialized agents using OpenAI models.
- Traces all tool, agent, and LLM calls using RagaAI Catalyst.

## Setup

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key
- A RagaAI Catalyst account

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/city-expert-agent.git
   cd city-expert-agent
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   RAGAAI_ACCESS_KEY=your_ragaai_access_key
   RAGAAI_SECRET_KEY=your_ragaai_secret_key
   ```

### Usage

1. Run the script:

   ```bash
   python llama.py
   ```

2. The script will fetch Wikipedia data, build indexes, and create agents. It will then use the top agent to answer a sample query about Boston.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.