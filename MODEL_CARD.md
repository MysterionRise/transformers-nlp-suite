# Model Card

## System

Transformers NLP Suite combines open-source NLP models behind a FastAPI service for customer-intelligence use cases. The platform performs sentiment analysis, named entity recognition, semantic similarity, extractive question answering, and summarization.

## Intended Use

- Analyze customer reviews, support notes, and survey feedback.
- Extract aggregate sentiment and frequently mentioned entities.
- Detect related or duplicate feedback through semantic similarity.
- Produce draft summaries for human review.
- Demonstrate production-minded AI engineering patterns in a portfolio setting.

## Out Of Scope

- Automated high-stakes decisions.
- Medical, legal, credit, employment, insurance, or safety-critical use.
- Fully autonomous customer messaging.
- Surveillance or profiling of individuals.
- Regulated production use without domain-specific evaluation and risk review.

## Default API Models

| Task | Registry key | Model ID |
| --- | --- | --- |
| Sentiment | `twitter_roberta_multilingual` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| NER | `spacy_sm` | `en_core_web_sm` |
| Similarity | `all_minilm_l6` | `sentence-transformers/all-MiniLM-L6-v2` |
| Summarization | `bart_large_cnn` | `facebook/bart-large-cnn` |
| QA | `distilbert_squad` | `distilbert-base-cased-distilled-squad` |

Additional model keys are available in `config/models.yaml`.

## Known Limitations

- Outputs can be wrong, biased, stale, or inconsistent across domains.
- Mixed sentiment text is reduced to a single primary label by the API.
- Entity extraction may miss domain-specific names or over-detect common nouns depending on the model.
- Generated summaries may omit important details or introduce misleading emphasis.
- Extractive QA can only answer questions supported by the supplied context.
- Latency depends on model size, hardware, cache warmth, network access, and memory pressure.

## Data And Privacy

The repo includes small sample/eval data only. Real deployments must define retention, redaction, access control, and logging policies before processing sensitive customer data.

The current structured request logs avoid raw body logging by default. Operators should keep debug logging disabled for sensitive environments.

## Evaluation

See `EVAL_REPORT.md` and `data/customer_intelligence_eval.json`.

The current eval set validates wiring and basic behavior. It does not establish production readiness for any domain.
