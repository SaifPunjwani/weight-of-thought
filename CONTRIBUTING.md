# Contributing to Weight-of-Thought

Thank you for your interest in contributing to the Weight-of-Thought project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [spunjwani6@gatech.edu].

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes
6. Submit a pull request

## Development Environment

### Setting Up

```bash
# Clone your fork
git clone https://github.com/yourusername/weight-of-thought.git
cd weight-of-thought

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=wot
```

## Contribution Workflow

1. **Find an issue to work on**: Look for issues labeled "good first issue" or "help wanted"
2. **Create a new branch**: Create a branch with a descriptive name for your changes
3. **Make your changes**: Implement your changes, following our coding standards
4. **Test your changes**: Ensure all tests pass and add new tests for your changes
5. **Document your changes**: Update documentation to reflect your changes
6. **Submit a pull request**: Push your changes and create a pull request

## Coding Standards

We follow these coding standards:

- Use [Black](https://black.readthedocs.io/) for code formatting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints where appropriate
- Write docstrings in Google style format
- Keep functions concise and focused on a single responsibility
- Use meaningful variable and function names

## Testing

- Write unit tests for all new functionality
- Ensure existing tests pass with your changes
- Aim for high test coverage (>80%)
- Include both positive and negative test cases

## Documentation

- Update docstrings for modified or new functions/classes
- Update README.md if necessary
- Add examples for new functionality
- Update any affected documentation files

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Fill out the pull request template completely
4. Reference any related issues
5. Wait for review from maintainers
6. Address any feedback from reviewers
7. Your contribution will be merged once approved

## Community

- Join our [Discussion Forum](https://github.com/SaifPunjwani/weight-of-thought/discussions)
- Participate in discussions on open issues
- Help others with their questions

Thank you for contributing to Weight-of-Thought!