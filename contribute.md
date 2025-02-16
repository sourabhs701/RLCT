# Contributing to RL Crypto Trading

Thank you for your interest in contributing to this project! This document provides guidelines and conventions for contributing.

## Commit Message Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Each commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

### Types
- **feat**: A new feature
  - Example: `feat: add TimeGAN model`
  - Example: `feat(data-pipeline): implement Binance data fetcher`

- **fix**: A bug fix
  - Example: `fix: correct data normalization in preprocessing`
  - Example: `fix(rl-agent): resolve memory leak in training loop`

- **docs**: Documentation changes
  - Example: `docs: update API documentation`
  - Example: `docs(readme): add installation instructions`

- **refactor**: Code changes that neither fix bugs nor add features
  - Example: `refactor: optimize data preprocessing pipeline`
  - Example: `refactor(env): simplify reward calculation`

- **test**: Adding or modifying tests
  - Example: `test: add unit tests for TimeGAN`
  - Example: `test(trading-env): add integration tests`

- **chore**: Changes to build process, tools, etc.
  - Example: `chore: update dependencies`
  - Example: `chore(ci): configure GitHub Actions`

## Pull Request Process

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b feature/your-feature
   ```

2. Make your changes:
   - Follow the commit message conventions
   - Keep commits focused and atomic
   - Include tests for new features

3. Update documentation:
   - Add or update docstrings
   - Update README.md if needed
   - Update API documentation if applicable

4. Submit Pull Request:
   - Fill out the PR template
   - Link related issues
   - Request reviews from relevant team members

## Code Style Guidelines

1. Python Code:
   - Follow PEP 8 guidelines
   - Use type hints for function arguments and return values
   - Maximum line length: 88 characters (Black formatter)

2. Documentation:
   - Use Google-style docstrings
   - Include examples in docstrings where appropriate
   - Keep documentation up to date with code changes

## Development Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/macOS
   venv\Scripts\activate     # Windows
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_timegan.py

# Run with coverage report
pytest --cov=src
```

## Code Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] No unnecessary code changes
- [ ] Error handling is appropriate
- [ ] Performance implications considered

## Questions or Problems?

- Create an issue for feature requests or bug reports
- Tag maintainers for urgent issues
- Join our Discord/Slack channel for discussions

## License

By contributing, you agree that your contributions will be licensed under the project's license.