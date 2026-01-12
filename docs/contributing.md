# Contributing

Thank you for your interest in NeuroHear! Contributions of all forms are welcome.

## How to Contribute

### Report Bugs

Submit bug reports in [Issues](https://github.com/neurohear/neurohear/issues), please include:
- Problem description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment info (Python version, PyTorch version, OS)

### Feature Suggestions

Discuss new feature ideas in [Discussions](https://github.com/neurohear/neurohear/discussions).

### Submit Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push branch: `git push origin feature/your-feature`
5. Create a Pull Request

## Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/neurohear.git
cd neurohear

# Install development dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Code formatting
uv run ruff format .
uv run ruff check --fix .
```

## Code Standards

- Use [ruff](https://github.com/astral-sh/ruff) for code formatting and linting
- Follow PEP 8 style guide
- Write docstrings for public APIs
- Add appropriate type annotations

## Testing

- Add tests for new features
- Ensure all tests pass: `uv run pytest`
- Maintain test coverage

## Documentation

- Update relevant documentation
- Add examples for new features
- Keep README and docs/ in sync

## Code of Conduct

Please communicate with others kindly and respectfully. We are committed to maintaining an open and inclusive community environment.
