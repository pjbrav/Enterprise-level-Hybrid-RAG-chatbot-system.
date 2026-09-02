# Hybrid RAG — Multi-Source Enterprise Search

A Hybrid Retrieval-Augmented Generation system that combines FTS5 lexical search,
entity knowledge graph retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder
reranking, with partition-based routing for multi-source enterprise data.

## Architecture

```
Query → spaCy NER → Partition Routing → FTS5 (BM25) + Entity Graph Search
                                        ↓
                                   RRF Fusion
                                        ↓
                              Cross-encoder Reranking
                                        ↓
                              Top Chunks → LLM → Answer
```

**Standard RAG:** Query → FTS5 search → top chunks → LLM → answer

**Hybrid RAG:** Query → entity extraction → partition routing → per-partition
FTS5 + entity graph → RRF fusion → cross-encoder reranking → LLM → answer

## Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Lexical search | SQLite FTS5 (BM25) | Term-frequency ranking |
| Entity extraction | spaCy NER + regex IDs | Named entity → chunk mapping |
| Entity graph | SQLite (34M rows) | Cross-partition entity matching |
| Rank fusion | RRF (k=60) | Merge lexical + entity rankings |
| Reranking | cross-encoder/ms-marco-TinyBERT | Semantic relevance scoring |
| Smart router | Confidence-based | Skip Hybrid when Standard is sufficient |
| Routing config | YAML | Per-deployment partition keywords |

## Results (500-question evaluation)

| Metric | Standard | Hybrid | Difference |
|--------|----------|--------|------------|
| Answer relevance | 65.2% | 66.1% | +0.9 pp |
| Token usage | 459,014 | 423,874 | -7.7% |
| Latency | 9.4s | 17.1s | +7.7s |
| Win rate (answer rel) | 36% | 39% | +3 pp |

Hybrid wins 39% of questions vs Standard's 36%, while saving 7.7% tokens.

## Dataset

- 1,924,473 chunks across 9 enterprise data sources
- 34M entity rows in the knowledge graph
- 500 evaluation questions across 10 question types
- Sources: Slack, Gmail, GitHub, Jira, Confluence, HubSpot, Linear, Fireflies, Google Drive

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set your API key
export OPENAI_API_KEY=sk-your-key-here

# Run a query
python -c "from v24 import build_large_corpus_engine; db,p,e = build_large_corpus_engine(); print(e.compare('What GitHub PRs relate to Jira tickets?'))"

# Run evaluation
python batch_compare_v2.py --force-hybrid --sample 50
```

## Configuration

All thresholds are environment-variable configurable (see `.env.example`).
Per-deployment routing keywords go in `routing_config.yaml`.

## Files

| File | Purpose |
|------|---------|
| `v24.py` | Main RAG system (FTS5 + entity graph + RRF + cross-encoder + smart router) |
| `batch_compare_v2.py` | Evaluation script (CLI args, file picker, CSV/MD output) |
| `test_v24.py` | Unit tests |
| `routing_config.yaml` | Per-deployment partition routing keywords |
| `requirements.txt` | Python dependencies |
| `setup.py` | Package installation |

## License

MIT
