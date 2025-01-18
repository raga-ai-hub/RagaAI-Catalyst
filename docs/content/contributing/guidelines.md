# Contribution Guidelines

## Code Standards

### Python Style Guide
- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters (Black default)
- Use docstrings for all public functions/classes

### Example
```python
from typing import Dict, Optional

def process_data(
    data: Dict[str, any],
    config: Optional[Dict[str, any]] = None
) -> Dict[str, any]:
    """
    Process input data according to configuration.

    Args:
        data: Input data dictionary
        config: Optional configuration parameters

    Returns:
        Processed data dictionary
    """
    # Implementation
    pass
```

## Documentation
- Update relevant documentation
- Add docstrings to new code
- Include examples for new features
- Update changelog

## Testing
- Write unit tests for new features
- Maintain or improve coverage
- Include integration tests when needed

## Git Workflow
1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## Pull Request Process
1. Update documentation
2. Add tests
3. Run linting and formatting
4. Update changelog
5. Request review