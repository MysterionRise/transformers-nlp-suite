# Transformers NLP Suite

Enterprise NLP Platform with production REST API, authentication, rate limiting, observability, and interactive UIs. Built with Hugging Face Transformers, FastAPI, and Gradio.

[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/MysterionRise/transformers-nlp-suite/network/updates)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

### Enterprise REST API (NEW!)
- 🚀 **Production FastAPI Server** - Full REST API with OpenAPI/Swagger documentation
- 🔐 **Authentication** - API key and JWT token support
- ⏱️ **Rate Limiting** - Per-user rate limits with configurable tiers
- 📊 **Prometheus Metrics** - Full observability with `/metrics` endpoint
- 📚 **Interactive Docs** - Swagger UI at `/docs` and ReDoc at `/redoc`

### Interactive UIs
- 🎭 **Sentiment Analysis Playground** - Analyze sentiment with multiple models
- 🔍 **Sentence Similarity Explorer** - Compare embeddings and semantic similarity
- 🏷️ **NER Visualizer** - Extract and visualize named entities
- 📝 **Text Summarization Studio** - Generate and compare text summaries
- 📊 **Model Performance Dashboard** - Compare and evaluate model performance

### Production-Ready Infrastructure
- ⚙️ **Centralized Configuration** - YAML-based model registry with 25+ pre-configured models
- 🔄 **Smart Model Caching** - LRU cache with automatic GPU/CPU detection
- 📝 **Structured Logging** - Colored console output with file rotation
- 🛡️ **Error Handling** - Custom exceptions and graceful degradation
- 🚀 **Performance Optimization** - Singleton patterns and lazy loading

### CLI Examples
- Sentiment analysis on reviews
- Named entity recognition with Spacy
- Text summarization with multiple models
- Sentence embeddings and similarity
- Data scraping from Google Play
- Comprehensive evaluation metrics (BLEU, ROUGE, METEOR, BERTScore)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MysterionRise/transformers-nlp-suite.git
cd transformers-nlp-suite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For NER functionality, download Spacy model:
```bash
python -m spacy download en_core_web_sm
# Or for better accuracy (larger model):
python -m spacy download en_core_web_trf
```

### Launch Interactive UIs

**Option 1: Interactive Menu**
```bash
python launch_ui.py
```

**Option 2: Launch Specific UI**
```bash
python launch_ui.py sentiment       # Sentiment Analysis
python launch_ui.py similarity      # Sentence Similarity
python launch_ui.py ner             # Named Entity Recognition
python launch_ui.py summarization   # Text Summarization
python launch_ui.py performance     # Model Performance Dashboard
```

**Option 3: Direct Launch**
```bash
python ui/sentiment_playground.py      # Port 7860
python ui/similarity_explorer.py       # Port 7861
python ui/ner_visualizer.py           # Port 7862
python ui/summarization_studio.py     # Port 7863
python ui/performance_dashboard.py    # Port 7864
```

### Launch REST API Server

```bash
# Launch API server
python launch_ui.py api

# Launch with auto-reload (development)
python launch_ui.py api --reload

# Custom host/port
python launch_ui.py api --host 0.0.0.0 --port 8080
```

After launching, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 📚 Interactive UIs Overview

### 🎭 Sentiment Analysis Playground
**Port:** 7860 | **File:** `ui/sentiment_playground.py`

Analyze text sentiment using state-of-the-art transformer models.

**Features:**
- Multiple model comparison (RoBERTa, DistilBERT, BERT)
- Real-time sentiment classification
- Confidence scores with visual bars
- Batch processing support
- Example texts included

**Models:**
- Twitter RoBERTa (Multilingual)
- Twitter RoBERTa (English)
- DistilBERT SST-2
- BERT Base SST-2

---

### 🔍 Sentence Similarity Explorer
**Port:** 7861 | **File:** `ui/similarity_explorer.py`

Explore semantic similarity between sentences using embeddings.

**Features:**
- Pairwise sentence comparison
- Semantic search (find most similar sentences)
- 2D embedding visualization (t-SNE, PCA)
- Similarity heatmaps
- Interactive plots

**Use Cases:**
- Duplicate detection
- Semantic search
- Content clustering
- Paraphrase detection

---

### 🏷️ NER Visualizer
**Port:** 7862 | **File:** `ui/ner_visualizer.py`

Extract and visualize named entities from text using Spacy.

**Features:**
- Interactive entity highlighting
- Entity type filtering (18+ entity types)
- Entity statistics and charts
- JSON export
- Pre-loaded examples

**Entity Types:**
- PERSON, ORG, GPE, LOC, DATE, TIME
- MONEY, PERCENT, PRODUCT, EVENT
- And more...

---

### 📝 Text Summarization Studio
**Port:** 7863 | **File:** `ui/summarization_studio.py`

Generate concise summaries with multiple transformer models.

**Features:**
- Single model summarization
- Multi-model comparison
- Adjustable parameters (min/max length)
- Summary statistics (compression ratio)
- Example articles

**Models:**
- BART Large CNN
- T5 Large/Base
- Pegasus XSum
- DistilBART CNN

---

### 📊 Model Performance Dashboard
**Port:** 7864 | **File:** `ui/performance_dashboard.py`

Compare and evaluate model outputs using comprehensive NLP metrics.

**Features:**
- Single and batch text comparison
- Multiple evaluation metrics (BLEU, ROUGE, METEOR, BERTScore, Cosine Similarity)
- Radar and bar chart visualizations
- Model registry browser
- Cache statistics viewer
- JSON export functionality

**Use Cases:**
- Model evaluation and comparison
- Quality assessment of generated text
- Benchmarking different models
- Research and experimentation

---

## 🚀 REST API

The Enterprise NLP API provides production-ready REST endpoints for all NLP tasks.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sentiment` | POST | Analyze text sentiment |
| `/api/v1/sentiment/batch` | POST | Batch sentiment analysis |
| `/api/v1/summarize` | POST | Generate text summary |
| `/api/v1/ner` | POST | Extract named entities |
| `/api/v1/similarity` | POST | Compute text similarity |
| `/api/v1/qa` | POST | Question answering |
| `/api/v1/models` | GET | List available models |
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |

### Authentication

All API endpoints require authentication via:

**API Key** (recommended for scripts):
```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

**JWT Token** (for web apps):
```bash
# Get token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "X-API-Key: dev-api-key" | jq -r '.access_token')

# Use token
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

### Example API Calls

**Sentiment Analysis:**
```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product exceeded all my expectations!",
    "model": "twitter_roberta_multilingual"
  }'
```

**Text Summarization:**
```bash
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has transformed numerous industries...",
    "min_length": 30,
    "max_length": 100
  }'
```

**Named Entity Recognition:**
```bash
curl -X POST http://localhost:8000/api/v1/ner \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple CEO Tim Cook announced new products in Cupertino.",
    "entity_types": ["PERSON", "ORG", "GPE"]
  }'
```

**Semantic Similarity:**
```bash
curl -X POST http://localhost:8000/api/v1/similarity \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "The weather is beautiful today.",
    "text2": "It is a lovely sunny day outside."
  }'
```

**Question Answering:**
```bash
curl -X POST http://localhost:8000/api/v1/qa \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "context": "France is a country in Western Europe. Its capital is Paris."
  }'
```

### Rate Limiting

Requests are rate-limited based on your API key tier:
- **Admin**: 1000 requests/minute
- **User**: 100 requests/minute
- **Demo**: 20 requests/minute

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when window resets

---

## 🏗️ Infrastructure Components

### Configuration Management (`config/`)
Centralized configuration system with YAML-based model registry:
- **models.yaml**: 25+ pre-configured models across 8 categories
- **settings.py**: Application settings with environment variable support
- Pydantic validation for type safety

```python
from config import get_model_registry, get_settings

# Access model configurations
registry = get_model_registry()
model_config = registry.get_model('sentiment_analysis', 'twitter_roberta_multilingual')

# Access settings
settings = get_settings()
print(f"Device: {settings.device}, Max cached models: {settings.max_cached_models}")
```

### Model Cache (`utils/model_cache.py`)
Intelligent model caching with LRU eviction:
- Automatic GPU/CPU/MPS detection
- Lazy loading (models loaded on first use)
- Memory management with automatic cleanup
- Thread-safe singleton pattern

```python
from utils import load_model

# Models are cached automatically
model = load_model('sentiment_analysis', 'twitter_roberta_multilingual')
```

### Error Handling (`utils/error_handler.py`)
Custom exceptions and decorators for robust error handling:
- `ModelLoadError`, `InferenceError`, `InvalidInputError`, etc.
- `@handle_errors` decorator for graceful degradation
- `@retry_on_error` decorator for transient failures
- User-friendly error messages

### Logging (`utils/logger.py`)
Structured logging with colored console output:
- Automatic log rotation
- Performance tracking with `PerformanceLogger`
- Configurable log levels
- File and console handlers

```python
from utils import get_logger, PerformanceLogger

logger = get_logger(__name__)
with PerformanceLogger("model_inference", logger=logger):
    result = model.predict(text)
```

---

## 🛠️ CLI Examples

### Sentiment Analysis
```bash
python sentiment-analysis.py
```
Analyzes sentiment of reviews from `data/all_reviews.csv`.

### Named Entity Recognition
```bash
python ner.py
```
Extracts entities from text and generates visualizations.

### Text Summarization
```bash
python summarisation_llm_test.py
```
Compares 5 summarization models on JSON files in `data/` directory.

### Sentence Embeddings
```bash
python embeddings_test.py
```
Generates sentence embeddings using sentence-transformers.

### Data Scraping
```bash
python scrap_reviews.py
```
Scrapes app reviews from Google Play Store.

### Evaluation
```bash
python eval/eval.py
```
Comprehensive NLP evaluation with BLEU, ROUGE, METEOR, BERTScore.

## 📦 Requirements

### Core Dependencies
- transformers[torch] - Hugging Face Transformers
- torch - PyTorch
- spacy - Industrial NLP
- numpy - Numerical computing

### UI Dependencies
- gradio>=4.0.0 - Interactive UIs
- plotly>=5.17.0 - Visualizations
- scikit-learn - ML utilities
- umap-learn - Dimensionality reduction

### Evaluation
- nltk - BLEU scores
- rouge-score - ROUGE metrics
- bert-score - BERT-based evaluation

### API Dependencies
- fastapi>=0.109.0 - REST API framework
- uvicorn[standard] - ASGI server
- python-jose[cryptography] - JWT authentication
- slowapi - Rate limiting
- prometheus-client - Metrics

See `requirements.txt` for full list.

## 🎓 Use Cases

- **Education**: Learn NLP concepts interactively
- **Prototyping**: Quickly test transformer models
- **Research**: Compare model performance
- **Development**: Build NLP applications
- **Demo**: Showcase NLP capabilities

## 📖 Additional Resources

- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Detailed feature roadmap
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [Spacy Documentation](https://spacy.io/usage)
- [Gradio Documentation](https://gradio.app/docs/)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by "Natural Language Processing with Transformers" book
- Built with Hugging Face Transformers
- UI powered by Gradio
- NER powered by Spacy
