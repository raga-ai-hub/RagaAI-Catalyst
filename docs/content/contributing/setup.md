# Development Setup

## Initial Setup

1. Clone the Repository
```bash
git clone https://github.com/raga-ai-hub/agentneo.git
cd agentneo
```

2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix
# or
venv\Scripts\activate  # Windows
```

3. Install Development Dependencies
```bash
pip install -e ".[dev]"
```

## Development Tools

### Required Tools
- Python 3.9+
- Git
- pytest
- black (code formatting)
- isort (import sorting)
- flake8 (linting)

### Installing Development Tools
```bash
pip install black isort flake8 pytest pytest-cov
```

## Setting Up Pre-commit Hooks
```bash
pre-commit install
```

## Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentneo tests/

# Run specific test file
pytest tests/test_tracer.py
```