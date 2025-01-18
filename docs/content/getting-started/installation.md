# Installation Guide

## Prerequisites
- Python 3.9 or above
- pip, Conda, or Pipenv (latest version recommended)

## Quick Installation
```bash
pip install agentneo
```

## Platform-Specific Installation

### Windows
```bash
# Create virtual environment
python -m venv agentneo-env
agentneo-env\Scripts\activate

# Install AgentNeo
pip install agentneo
```

### macOS/Linux
```bash
# Create virtual environment
python -m venv agentneo-env
source agentneo-env/bin/activate

# Install AgentNeo
pip install agentneo
```

### Using Conda
```bash
# Create conda environment
conda create --name agentneo-env python=3.10
conda activate agentneo-env

# Install AgentNeo
pip install agentneo
```

## Verification
Verify your installation:
```python
python -c "import agentneo; print(agentneo.__version__)"
```

## Development Installation
For contributing or development:
```bash
git clone https://github.com/raga-ai-hub/agentneo.git
cd agentneo
pip install -e ".[dev]"
```

## Next Steps
- [Quick Start Guide](quick-start.md)
- [Basic Usage](../core-concepts/basic-usage.md)