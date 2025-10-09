# Contributing to Structured Document Parser

Thank you for your interest in contributing to the Structured Document Parser! This document provides guidelines and instructions for contributors.

## Development Setup

1. Fork the repository and clone it locally.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install development dependencies:
   ```bash
   pip install pytest black flake8 mypy
   ```

## Project Structure

```
structured_parser/
├── src/
│   ├── structured_extraction.py  # Main entry point
│   ├── utils.py                  # Utility functions
│   ├── typedicts.py              # Type definitions
│   ├── json_to_sql.py            # Database integration
│   └── config.yaml               # Configuration
├── tests/                        # Test cases
├── README.md                     # Project overview
└── requirements.txt              # Dependencies
```

## Development Workflow

### Code Style

We follow the PEP 8 style guide for Python code. Please use `black` for formatting:

```bash
black src/
```

### Type Checking

We use type hints and `mypy` for type checking:

```bash
mypy src/
```

### Testing

Please add tests for new features and ensure all tests pass:

```bash
pytest tests/
```

## Areas for Contribution

Here are some areas where contributions are especially welcome:

### 1. Artifact Extraction Improvements

- Adding support for new artifact types
- Improving extraction accuracy for existing types
- Optimizing prompts for better results

### 2. Performance Optimization

- Improving inference speed
- Reducing memory usage
- Implementing efficient batching strategies

### 3. New Features

- Supporting additional document types (beyond PDF)
- Adding new output formats
- Implementing document comparison functionality
- Enhancing vector search capabilities

### 4. Documentation and Examples

- Improving documentation
- Adding usage examples
- Creating tutorials or guides

## Submitting Changes

1. Create a new branch for your changes
2. Make your changes and commit with clear commit messages
3. Push your branch and submit a pull request
4. Ensure CI tests pass

## Pull Request Guidelines

- Provide a clear description of the problem and solution
- Include any relevant issue numbers
- Add tests for new functionality
- Update documentation as needed
- Keep pull requests focused on a single topic

## Prompt Engineering Guidelines

When modifying prompts in `config.yaml`, consider:

1. **Clarity**: Provide clear and specific instructions
2. **Examples**: Include examples where helpful
3. **Structure**: Use structured formatting to guide the model
4. **Schema alignment**: Ensure prompts align with output schemas
5. **Testing**: Test prompts with diverse document types

## Output Schema Guidelines

When defining output schemas:

1. Keep properties focused and well-defined
2. Use descriptive field names
3. Include descriptions for complex fields
4. Consider required vs. optional fields carefully
5. Test schemas with different document layouts

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions or Issues?

If you have questions or encounter issues, please:

1. Check existing issues to see if it's been addressed
2. Open a new issue with a clear description and steps to reproduce
3. Tag relevant project maintainers

Thank you for your contributions!
