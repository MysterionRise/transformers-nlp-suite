# Transformers NLP Suite - Claude Code Guide

This file provides guidance to Claude Code (claude.ai/code) when working with the Transformers NLP Suite repository.

## Quick Reference

### Essential Commands

**Setup:**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development and testing
python -m spacy download en_core_web_sm  # For NER functionality
```

**Testing:**
```bash
pytest                           # Run all tests
pytest -m "not slow"             # Run fast tests only
pytest -m slow                   # Run integration tests only
pytest tests/unit/test_config.py # Run a specific test file
pytest -k test_name              # Run tests matching pattern
pytest -v --tb=short             # Verbose output with short tracebacks
```

**Code Quality:**
```bash
black .                          # Format code
isort .                          # Sort imports
flake8 . --max-line-length=120   # Check code style
pytest --cov=. --cov-report=html # Generate coverage report
```

**UI Launches:**
```bash
python launch_ui.py              # Interactive menu
python launch_ui.py sentiment    # Specific UI
python ui/sentiment_playground.py # Direct launch (port 7860)
```

### Python Version
- **Required:** Python 3.11+
- **CI tests:** 3.11, 3.12, 3.13
- Pre-commit hooks require matching Python version

## Architecture Overview

### Core Configuration System (`config/`)
The project uses a **centralized, Pydantic-validated configuration system** with YAML-based model registry:

- **`settings.py`**: Defines `Settings` (app configuration) and `ModelRegistry` (model definitions)
- **`models.yaml`**: YAML file with 25+ pre-configured models across 8 categories (sentiment_analysis, summarization, embeddings, ner, etc.)
- **Singleton pattern**: `get_settings()` and `get_model_registry()` use LRU caching to return cached instances
- **Environment variable support**: Settings can be overridden with `NLP_*` prefix (e.g., `NLP_DEBUG=true`, `NLP_MAX_CACHED_MODELS=5`)

**Usage pattern:**
```python
from config import get_settings, get_model_registry

settings = get_settings()  # Access app settings
registry = get_model_registry()  # Access model configs
model_config = registry.get_model('sentiment_analysis', 'twitter_roberta_multilingual')
```

### Model Caching Layer (`utils/model_cache.py`)
**Singleton instance** that handles intelligent model loading and memory management:

- **Lazy loading**: Models load on first use, not at startup
- **LRU cache**: Automatically evicts least-recently-used models when cache is full (default: 3 models)
- **Device detection**: Automatically selects CUDA > MPS > CPU
- **Thread-safe**: Uses threading locks for concurrent access
- **Memory cleanup**: Calls `gc.collect()` and `torch.cuda.empty_cache()` on eviction
- **Retry logic**: `@retry_on_error` decorator on `_load_model()` for transient failures

**Public API:**
- `get_model_cache()`: Get singleton instance
- `load_model(category, model_key)`: Load from registry
- `load_model_by_id(model_id, task)`: Load by HuggingFace ID directly
- `get_cache_info()`: Get cache statistics
- `clear_model_cache()`: Clear all cached models

### Error Handling (`utils/error_handler.py`)
Custom exception hierarchy with decorators:
- `ModelLoadError`, `ModelNotFoundError`, `InferenceError`, `InvalidInputError`
- `@handle_errors` decorator for graceful degradation with user-friendly messages
- `@retry_on_error` decorator with configurable retries and exponential backoff
- `ErrorContext` context manager for structured error tracking

### Logging (`utils/logger.py`)
Structured logging with performance tracking:
- `get_logger(name)`: Get module-level logger with colored console output
- `PerformanceLogger`: Context manager for timing code blocks and logging performance metrics
- Automatic log rotation to `logs/` directory
- Configurable log levels via `NLP_LOG_LEVEL` environment variable

## Project Structure

```
.
├── config/                          # Configuration management
│   ├── __init__.py
│   ├── settings.py                 # Settings and ModelRegistry classes
│   └── models.yaml                 # Model definitions (25+ models)
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── model_cache.py              # Model loading and LRU cache
│   ├── error_handler.py            # Custom exceptions and decorators
│   └── logger.py                   # Structured logging
│
├── ui/                             # Interactive Gradio UIs
│   ├── sentiment_playground.py     # Port 7860
│   ├── similarity_explorer.py      # Port 7861
│   ├── ner_visualizer.py           # Port 7862
│   ├── summarization_studio.py     # Port 7863
│   └── performance_dashboard.py    # Port 7864
│
├── tests/                          # Test suite
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
│
├── eval/                           # Evaluation utilities
├── examples/                       # Example scripts
├── launch_ui.py                    # Multi-UI launcher
├── pyproject.toml                  # Black, isort, pytest config
├── pytest.ini                      # Test configuration
└── requirements*.txt               # Dependencies
```

## Test Organization

Tests are structured in two categories marked with pytest markers:

**Unit Tests** (`tests/unit/`, `@pytest.mark.unit`):
- Fast, no external dependencies
- Test configuration, model cache, individual components
- Run with: `pytest -m "not slow"`

**Integration Tests** (`tests/integration/`, `@pytest.mark.slow`):
- Download actual models from HuggingFace, use Spacy
- Test full pipelines and UI components
- Run with: `pytest -m slow` (30-45 minute timeout in CI)

**Fixture setup** (`tests/conftest.py`):
- `sample_text`: Positive sentiment text
- `sample_negative_text`: Negative sentiment text
- `sample_sentences`: Four sentences for similarity testing
- `sample_long_text`: Multi-sentence article for summarization

## Code Style and Formatting

- **Line length:** 120 characters (configured in `pyproject.toml`)
- **Formatter:** Black with black-compatible isort
- **Linter:** Flake8 (max complexity 10, ignoring W605 and E203)
- **Pre-commit hooks:** Enforce isort, black, flake8, bandit (security check), YAML validation
- **Commit messages:** Must follow Commitizen format (enforced by pre-commit)

Run all checks with:
```bash
black . && isort . && flake8 . && pytest
```

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. **Lint and Format Check** (all PRs)
   - Black format check
   - isort check
   - Flake8 style and security checks

2. **Fast Tests** (all PRs, parallel across Python 3.11/3.12/3.13)
   - Runs unit tests with coverage
   - Skips slow/integration tests
   - Uploads coverage to Codecov

3. **Integration Tests** (all pushes/PRs, Python 3.11 only)
   - Runs slow tests with 30-minute timeout
   - Downloads Spacy model
   - Caches HuggingFace models between runs

4. **Full Test Suite** (all pushes, Python 3.11 only)
   - Runs all tests with coverage
   - Generates HTML coverage report
   - 45-minute timeout

## Key Dependencies

### Core NLP
- `transformers[torch]`: HuggingFace Transformers library
- `torch`: PyTorch backend
- `spacy`: Industrial-strength NLP
- `sentence-transformers`: Embeddings

### UI and Visualization
- `gradio>=4.0.0`: Web UIs
- `plotly>=5.17.0`: Interactive charts
- `scikit-learn`: ML utilities (embeddings visualization)
- `umap-learn`: Dimensionality reduction

### Evaluation
- `rouge-score`: ROUGE metrics for summarization
- `bert-score`: BERT-based text evaluation
- `nltk`: BLEU scores

### Configuration
- `pydantic>=2.4.0`: Settings validation
- `pydantic-settings>=2.0.0`: Environment variable support
- `pyyaml>=6.0.1`: YAML parsing
- `python-dotenv>=1.0.0`: .env file support

## Common Development Tasks

### Adding a New Model to the Registry
1. Edit `config/models.yaml` and add model under appropriate category
2. Ensure `name`, `model_id`, `task` fields are set
3. Optional fields: `min_length`, `max_length`, `dimensions`, `languages`, `prefix`
4. Add unit test in `tests/unit/test_config.py`

### Creating a New Interactive UI
1. Create file in `ui/` directory (e.g., `ui/new_task.py`)
2. Import from `config` and `utils` for consistency
3. Use `load_model()` from model cache
4. Create Gradio interface with `gr.Interface` or `gr.Blocks`
5. Add launcher option in `launch_ui.py`

### Adding Tests
- Unit tests go in `tests/unit/` with `@pytest.mark.unit`
- Integration tests in `tests/integration/` with `@pytest.mark.slow`
- Use fixtures from `conftest.py`
- Prefix test files with `test_` and functions with `test_`

### Debugging Model Loading Issues
1. Set `NLP_DEBUG=true` to enable debug logging
2. Check `logs/` directory for detailed output
3. Use `get_cache_info()` to inspect cache state
4. Verify model exists in `config/models.yaml`
5. Try `load_model_by_id(model_id, task)` to load by HuggingFace ID directly

## Pre-commit Setup

The repository uses pre-commit hooks for automated code quality checks:
```bash
pre-commit install              # Install hooks
pre-commit run --all-files      # Run all hooks manually
git commit                      # Hooks run automatically on commit
```

Hooks enforce:
- Import sorting (isort)
- Code formatting (black)
- Style checks (flake8)
- Security checks (bandit)
- YAML validation
- File formatting (trailing whitespace, line endings)
- Commit message format (commitizen)

## Environment Variables

Key variables that can override settings:
- `NLP_DEBUG=true/false`: Enable debug logging
- `NLP_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR`: Set logging level
- `NLP_DEVICE=auto|cpu|cuda|mps`: Force device selection
- `NLP_MAX_CACHED_MODELS=N`: Maximum cached models (default 3)
- `NLP_CACHE_DIR=/path`: HuggingFace model cache directory
- `NLP_SHARE=true`: Create public Gradio links

## Important Files and Their Purpose

- `config/models.yaml`: Single source of truth for all available models
- `config/settings.py`: Application settings and configuration validation
- `utils/model_cache.py`: Critical for performance - handles model lifecycle
- `.pre-commit-config.yaml`: Defines all quality gates that block commits
- `.github/workflows/ci.yml`: Full CI/CD logic
- `pytest.ini`: Test discovery and marker definitions
