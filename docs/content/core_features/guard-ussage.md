# LLM Guard Scanner Documentation

## Overview
This system provides protection against PII exposure and prompt injection in LLM interactions using various scanners. The system consists of input scanners that process prompts before they reach the LLM and output scanners that process the LLM's responses.

## Quick Start

```python
from openai import OpenAI
from agentneo import AgentNeo, Tracer

neo_session = AgentNeo(session_name="your_session_name")
neo_session.connect_project(project_name="your_project_name")

scanner_configs = {
    "input_scanners": {
        "Anonymize": {},
        "PromptInjection": {
            "threshold": 0.5,
            "match_type": "FULL"
        }
    },
    "output_scanners": {
        "Deanonymize": {}
    }
}

# Initialize tracer with guard
tracer = Tracer(
    session=neo_session,
    use_guard=True,
    llm_guard_config=scanner_configs
)

# Start tracing
tracer.start()

# Your api calls here...

# Stop tracing
tracer.stop()
```

## Available Scanners

### Input Scanners

#### 1. Anonymize Scanner
Detects and anonymizes personal identifiable information (PII) in the input text.

```python
"Anonymize": {
    "preamble": str,           # Optional prefix for anonymized entities
    "allowed_names": list,     # Names that should not be anonymized
    "hidden_names": list,      # Names that should always be anonymized
    "entity_types": list,      # Types of entities to anonymize
    "use_faker": bool,         # Use faker for replacement (default: False)
    "threshold": float,        # Confidence threshold (default: 0)
    "language": str           # Language code (default: "en")
}
```

#### 2. PromptInjection Scanner
Detects potential prompt injection attacks.

```python
"PromptInjection": {
    "threshold": float,        # Detection threshold (default: 0.5)
    "match_type": str         # Match type: "FULL" or other InjectionMatchType values
}
```

#### 3. Regex Scanner
Applies custom regex patterns for detection/redaction.

```python
"Regex": {
    "patterns": list,         # List of regex patterns
    "is_blocked": bool,       # Whether to block matched patterns
    "match_type": str,        # "SEARCH" or other RegexMatchType values
    "redact": bool           # Whether to redact matched content
}
```

Example patterns:
```python
patterns = [
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IP addresses
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", # Phone numbers
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b" # Email addresses
]
```

#### 4. Secrets Scanner
Detects sensitive information like API keys and credentials.

```python
"Secrets": {}  # No configuration required
```

### Output Scanners

#### 1. Deanonymize Scanner
Restores anonymized content in the output using the vault.

```python
"Deanonymize": {}  # No configuration required
```

## Common Configurations

### Basic PII Protection
```python
scanner_configs = {
    "input_scanners": {
        "Anonymize": {},
        "Secrets": {}
    },
    "output_scanners": {
        "Deanonymize": {}
    }
}
```

### Advanced Security
```python
scanner_configs = {
    "input_scanners": {
        "Anonymize": {
            "threshold": 0.8,
            "language": "en"
        },
        "PromptInjection": {
            "threshold": 0.6,
            "match_type": "FULL"
        },
        "Regex": {
            "patterns": [
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IP addresses
                r"\b\d{16}\b"                     # Credit card numbers
            ],
            "is_blocked": True,
            "match_type": "SEARCH",
            "redact": True
        },
        "Secrets": {}
    },
    "output_scanners": {
        "Deanonymize": {}
    }
}
```

## Entity Types for Anonymization
The Anonymize scanner can detect and replace various entity types:
- PERSON
- EMAIL
- PHONE
- ADDRESS
- CREDIT_CARD
- IP_ADDRESS
- KEY
- URL
- DOMAIN

## Error Handling
The system will raise a ValueError if scanner creation fails. The error message will include the scanner name and the specific error details.
