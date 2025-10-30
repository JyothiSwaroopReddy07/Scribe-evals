# Contributing Guidelines

Thank you for your interest in contributing to the SOAP Note Evaluation Framework. This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and professional in all interactions. We aim to create a welcoming environment for all contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/deepscribe-evals.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `make test`
6. Commit your changes: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# Run tests
make test

# Run linters
make lint

# Format code
make format
```

## Code Standards

### Python Style

- Follow PEP 8 guidelines
- Maximum line length: 120 characters
- Use type hints for all function signatures
- Write docstrings for all public functions and classes

### Documentation Standards

- Use Google-style docstrings
- Include examples in docstrings where helpful
- Update README.md if adding new features
- Update CHANGELOG.md with your changes

### Testing Standards

- Write unit tests for all new features
- Maintain test coverage above 80%
- Use descriptive test names
- Include both positive and negative test cases

## Pull Request Process

1. **Create Issue First**: For major changes, create an issue to discuss the proposed changes
2. **Update Tests**: Add or update tests for your changes
3. **Update Documentation**: Update relevant documentation
4. **Run Full Test Suite**: Ensure all tests pass
5. **Update CHANGELOG**: Add entry to CHANGELOG.md
6. **Code Review**: Address all review comments
7. **Squash Commits**: Squash commits into logical units before merging

## Adding New Evaluators

To add a new evaluator:

1. Create a new file in `src/evaluators/`
2. Inherit from `BaseEvaluator`
3. Implement the `evaluate` method
4. Add unit tests in `tests/`
5. Update documentation

Example:

```python
from src.evaluators.base_evaluator import BaseEvaluator, EvaluationResult

class MyEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__("MyEvaluator")
    
    def evaluate(self, transcript, generated_note, reference_note=None, note_id=""):
        # Implementation here
        score = compute_score(generated_note)
        issues = find_issues(generated_note)
        
        return EvaluationResult(
            note_id=note_id,
            evaluator_name=self.name,
            score=score,
            issues=issues,
            metrics={"custom_metric": value}
        )
```

## Adding New Data Sources

To add support for a new dataset:

1. Add a method to `DataLoader` class
2. Return list of `SOAPNote` objects
3. Handle errors gracefully
4. Add unit tests
5. Document in README.md

## Commit Message Format

Use clear, descriptive commit messages:

```
type: brief description

Longer description if needed

Fixes #issue-number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

## Release Process

1. Update version number in relevant files
2. Update CHANGELOG.md
3. Create a git tag: `git tag v1.x.x`
4. Push tag: `git push origin v1.x.x`
5. Create GitHub release with release notes

## Questions?

If you have questions, please:
1. Check existing documentation
2. Search existing issues
3. Create a new issue with your question

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

