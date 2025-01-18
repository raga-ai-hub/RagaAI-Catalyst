# Contributing to AgentNeo

First off, thank you for considering contributing to AgentNeo! It's people like you that make AgentNeo such a great tool. We welcome contributions from everyone, regardless of their experience level.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## How Can I Contribute?

### Reporting Bugs

- Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/raga-ai-hub/agentneo/issues).
- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/raga-ai-hub/agentneo/issues/new). Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

- Open a new issue with a clear title and detailed description of the suggested enhancement.
- Explain why this enhancement would be useful to most AgentNeo users.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Setting Up Your Development Environment

1. Ensure you have Python 3.9 or higher installed.
2. Install Node.js (version 14 or higher) and npm (version 6 or higher) or yarn (1.22+).
3. Fork the AgentNeo repository on GitHub.
4. Clone your fork locally:
   ```
   git clone https://github.com/your-username/agentneo.git
   cd agentneo
   ```
5. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
6. Install the development dependencies:
   ```
   pip install -e ".[dev]"
   ```

## Running Tests

We use pytest for our Python test suite. To run the tests:

```
pytest
```

For JavaScript tests, we use Jest. To run the JavaScript tests:

```
npm test
```

## Coding Style

### Python
We follow PEP 8 for Python code. Please ensure your code adheres to this standard. You can use tools like `flake8` or `black` to help format your code.

### JavaScript
For JavaScript code, we follow the Airbnb JavaScript Style Guide. We use ESLint to enforce these standards. Make sure to run ESLint before submitting your PR:

```
npm run lint
```

## Documentation

If you're adding a new feature, please update the documentation accordingly. This includes both in-code documentation (docstrings for Python, JSDoc for JavaScript) and any relevant updates to README.md or other documentation files.

## Roadmap Contributions

If you're working on a feature that's on our roadmap:

1. Check the current status in the README.md file.
2. If it's not already being worked on, feel free to take it on!
3. Update the roadmap in your PR to reflect the new status of the feature.

## Questions?

Don't hesitate to ask questions if something is unclear. You can open an issue for broader questions about the project.

Thank you for your contribution to AgentNeo! 🚀
