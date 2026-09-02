"""
Hybrid RAG v24 – Smart Router + spaCy NER + Auto‑Partitioning
==================================================================

- Smart router: runs Hybrid only when Standard confidence is low (force_hybrid=True for eval)
- Entity extraction: spaCy NER (PERSON, ORG, PRODUCT) + regex ID fallback (ticket IDs, PRs, commits)
- Automatically partitions documents using content‑based clustering when files are in a flat folder
- Falls back to folder‑based partitioning if documents are in subfolders
- Per-customer routing via routing_config.yaml (overrides hardcoded defaults)
- All thresholds env-configurable (os.getenv with sensible defaults)
- No dataset-specific stopwords – universal English only
- spaCy falls back to regex extractor if not installed (no hard dependency)
- Works with any document set — no overfitting to a specific corpus
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Environment & optional imports
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None

_EMBEDDING_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDING_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_PATH = os.getenv("SQLITE_DATABASE_PATH", "hybrid_rag_v22.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".py", ".js", ".ipynb"}
MAX_ZIP_FILES = int(os.getenv("MAX_ZIP_FILES", "250"))
MAX_ZIP_MEMBER_BYTES = int(os.getenv("MAX_ZIP_MEMBER_BYTES", str(25 * 1024 * 1024)))
# Folder names that carry no partitioning signal (e.g. a zip whose members
# all sit under "documents/"). Seeing only these names is treated the same
# as a flat/un-foldered corpus -- auto content-partitioning kicks in instead
# of literally creating a single partition called "documents".
GENERIC_FOLDER_NAMES = {
    "documents", "docs", "uploads", "papers", "files", "data", "corpus",
    "input", "inputs", "source", "sources", "pdfs", "pdf", "",
}
# Above this much extracted text, skip TF-IDF+KMeans clustering (it needs
# the whole corpus in memory at once) and fall back to a cheap per-document
# keyword label instead. Keeps a ~1.5GB raw upload (much smaller once PDFs
# are reduced to text) from stalling ingestion or blowing up memory.
AUTO_PARTITION_MAX_CORPUS_CHARS = int(os.getenv("AUTO_PARTITION_MAX_CORPUS_CHARS", str(80_000_000)))
PGVECTOR_DIMENSIONS = int(os.getenv("PGVECTOR_DIMENSIONS", "384"))
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
CHUNK_WORDS = 220
CHUNK_OVERLAP = 25
MAX_CHUNKS = 5
MAX_CROSS_DOMAIN_CHUNKS = 3
MIN_CHUNKS_PER_ACTIVE_PARTITION = 1
MAX_CROSS_DOMAIN_CHUNKS_CEILING = 6
MAX_ANSWER_TOKENS = 3000
CONTEXT_CHARS_PER_CHUNK = 1100

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")
CROSS_ENCODER_RERANK_TOP_K = 3

PROVIDER_PRICING_DEFAULTS = {
    "claude": (0.80, 2.40),
    "openai": (1.00, 6.00),
    "gemini": (0.15, 0.60),
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "for",
    "with", "on", "at", "from", "by", "in", "and", "or", "but", "as",
    "what", "why", "how", "when", "where", "who", "which", "does", "do",
    "compare", "comparison", "contrast", "explain", "describe", "analyze",
    "analyse", "evaluate", "between", "both", "versus", "than", "into",
}

# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------
def is_running_in_docker() -> bool:
    return os.path.exists('/.dockerenv')

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def estimate_tokens(text: str) -> int:
    if text and tiktoken is not None:
        try:
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            pass
    return math.ceil(len(text) / 4) if text else 0

def stem_token(token: str) -> str:
    token = token.lower().strip("-_")
    for suffix in ("izations", "ization", "ations", "ation", "ments", "ment", "ingly", "ing", "ies", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)] + ("y" if suffix == "ies" else "")
    return token

def query_terms(query: str) -> List[str]:
    return [
        stem_token(term) for term in re.findall(r"\b[\w-]+\b", query.lower())
        if len(term) > 2 and term not in STOPWORDS
    ]

# Universal English stopwords for the regex entity extractor fallback.
# With spaCy NER active, this list only affects the regex fallback path
# (used when spaCy is not installed). It contains only language-level
# stopwords — no dataset-specific template terms. Template boilerplate
# ("customer", "motivation", "checklist", etc.) was removed because:
#   1. spaCy doesn't extract these as entities (not PERSON/ORG/PRODUCT)
#   2. They are dataset-specific and don't generalize to new deployments
# Question words are kept because they pollute entity_search at scale
# regardless of dataset.
_ENTITY_STOPWORDS = frozenset({
    # Articles, pronouns, prepositions, conjunctions
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "done",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "for", "from", "with", "about", "into", "onto", "over", "under",
    "and", "or", "but", "if", "then", "else", "so", "as", "at", "by",
    "in", "on", "to", "of", "up", "out", "off", "down", "away",
    "yes", "no", "not", "nor", "yet", "ok", "okay",
    "you", "your", "yours", "we", "our", "ours", "us",
    "i", "me", "my", "mine", "they", "them", "their", "theirs",
    "he", "she", "his", "her", "hers", "him",
    "please", "thanks", "thank", "hi", "hey", "hello",
    # Question words — universal: "what" matches thousands of chunks
    # in any corpus, not just this one
    "what", "who", "whom", "whose", "which", "how", "when", "where", "why",
    "there", "here",
    # Time words — universal abbreviations capitalized in timestamps
    "utc", "est", "pst", "cet", "ist",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "today", "yesterday", "tomorrow",
})


_SPACY_NLP = None
_SPACY_AVAILABLE = False
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    pass

_SPACY_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT",
    "WORK_OF_ART", "LAW", "PRODUCT", "LOC",
}


def _get_spacy_nlp():
    """Lazily load spaCy model on first use (saves startup time)."""
    global _SPACY_NLP
    if _SPACY_NLP is None and _SPACY_AVAILABLE:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            try:
                _SPACY_NLP = spacy.load("en_core_web_md", disable=["parser", "lemmatizer"])
            except Exception:
                _SPACY_NLP = None
    return _SPACY_NLP


def _add_regex_ids(text: str, entities: Set[str]) -> None:
    """Extract structured IDs that spaCy misses."""
    # Ticket IDs: PROJ-123, ENG-4567
    for m in re.findall(r"\b([A-Z]{2,10}-\d{1,8})\b", text):
        entities.add(m.lower())
    # PR/issue references: #1234
    for m in re.findall(r"#(\d{2,8})\b", text):
        entities.add(f"#{m}")
    # Commit hashes: 7-40 hex chars
    for m in re.findall(r"\b([0-9a-f]{7,40})\b", text):
        if len(m) >= 7:
            entities.add(m)
    # Version numbers: v2.3.1, 2.0.0
    for m in re.findall(r"\b(v?\d+\.\d+\.\d+)\b", text):
        entities.add(m.lower())


_SPACY_NLP = None
_SPACY_AVAILABLE = False
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    pass

_SPACY_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT",
    "WORK_OF_ART", "LAW", "PRODUCT", "LOC",
}


def _get_spacy_nlp():
    """Lazily load spaCy model on first use (saves startup time)."""
    global _SPACY_NLP
    if _SPACY_NLP is None and _SPACY_AVAILABLE:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            try:
                _SPACY_NLP = spacy.load("en_core_web_md", disable=["parser", "lemmatizer"])
            except Exception:
                _SPACY_NLP = None
    return _SPACY_NLP


def _add_regex_ids(text: str, entities: Set[str]) -> None:
    """Extract structured IDs that spaCy misses."""
    # Ticket IDs: PROJ-123, ENG-4567
    for m in re.findall(r"\b([A-Z]{2,10}-\d{1,8})\b", text):
        entities.add(m.lower())
    # PR/issue references: #1234
    for m in re.findall(r"#(\d{2,8})\b", text):
        entities.add(f"#{m}")
    # Commit hashes: 7-40 hex chars
    for m in re.findall(r"\b([0-9a-f]{7,40})\b", text):
        if len(m) >= 7:
            entities.add(m)
    # Version numbers: v2.3.1, 2.0.0
    for m in re.findall(r"\b(v?\d+\.\d+\.\d+)\b", text):
        entities.add(m.lower())


def extract_entities(text: str) -> Set[str]:
    """Extract entities using spaCy NER + regex ID fallback.

    spaCy provides context-aware named entity recognition (PERSON, ORG,
    PRODUCT, etc.) that is far more accurate than regex for real names.
    A regex fallback catches structured IDs spaCy misses: ticket IDs,
    PR numbers, commit hashes.

    All entities are lowercased. If spaCy is not installed or the model
    is not downloaded, falls back to the original regex-only extractor.
    """
    entities = set()

    # --- spaCy NER path ---
    nlp = _get_spacy_nlp()
    if nlp is not None:
        truncated = text[:50000] if len(text) > 50000 else text
        doc = nlp(truncated)
        for ent in doc.ents:
            if ent.label_ in _SPACY_ENTITY_TYPES:
                name = ent.text.strip().lower()
                if len(name) >= 3 and name not in _ENTITY_STOPWORDS:
                    entities.add(name)
        _add_regex_ids(text, entities)
        return entities

    # --- Regex fallback (spaCy not available) ---
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*)*\b", text)
    for c in candidates:
        c = c.strip().lower()
        if len(c) >= 3 and c not in _ENTITY_STOPWORDS:
            entities.add(c)
    number_pattern = r"\b[\$€]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:psi|USD|EUR|bar|%|k|M)?\b"
    for num in re.findall(number_pattern, text):
        num = num.strip()
        if num and len(num) > 2:
            entities.add(num)
    id_pattern = r"\b\d{4,}\b"
    for id_val in re.findall(id_pattern, text):
        if len(id_val) == 4 and 1900 <= int(id_val) <= 2099:
            continue
        entities.add(id_val)
    _add_regex_ids(text, entities)
    return {e for e in entities if len(e) > 2}
# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
@dataclass
class ChunkMetadata:
    chunk_id: str
    file_path: str
    partition_id: str
    chunk_index: int
    token_count: int
    entity_count: int


class DatabaseManager:
    CREDIT_PROVIDERS = ("claude", "openai", "gemini")

    def __init__(self, db_path: str = DATABASE_PATH, database_url: str = None):
        self.db_path = db_path
        if database_url is None:
            database_url = os.getenv("DATABASE_URL")
        if database_url and is_running_in_docker():
            database_url = database_url.replace("localhost", "host.docker.internal")
        self.database_url = (database_url or "").strip()
        self.use_postgres = bool(self.database_url)
        self.pgvector_enabled = False
        if self.use_postgres and psycopg2 is None:
            raise RuntimeError("DATABASE_URL is set but psycopg2-binary is not installed.")
        self._init_db()

    @property
    def backend_name(self) -> str:
        return "PostgreSQL + pgvector" if self.use_postgres else "SQLite"

    def _connect(self) -> Any:
        if self.use_postgres:
            return psycopg2.connect(self.database_url)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, query: str, params: Sequence[Any] = (), fetch: bool = False) -> List[Dict[str, Any]]:
        conn = self._connect(); conn.execute("PRAGMA journal_mode=WAL")
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor) if self.use_postgres else conn.cursor()
            cursor.execute(query.replace("?", "%s") if self.use_postgres else query, tuple(params))
            rows = [dict(row) for row in cursor.fetchall()] if fetch else []
            conn.commit()
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _many(self, query: str, records: Sequence[Sequence[Any]]) -> None:
        if not records:
            return
        conn = self._connect(); conn.execute("PRAGMA journal_mode=WAL")
        try:
            cursor = conn.cursor()
            cursor.executemany(query.replace("?", "%s") if self.use_postgres else query, records)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        if self.use_postgres:
            try:
                self._query("CREATE EXTENSION IF NOT EXISTS vector")
                self.pgvector_enabled = True
            except Exception as exc:
                print(f"pgvector unavailable; storing embeddings as JSON: {exc}")
            vector_column = f"VECTOR({PGVECTOR_DIMENSIONS})" if self.pgvector_enabled else "TEXT"
            statements = [
                """CREATE TABLE IF NOT EXISTS documents (file_path TEXT PRIMARY KEY, file_name TEXT NOT NULL, partition_id TEXT NOT NULL, processed_date TEXT NOT NULL, chunk_count INTEGER NOT NULL, content_hash TEXT)""",
                f"""CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, partition_id TEXT NOT NULL, chunk_text TEXT NOT NULL, token_count INTEGER NOT NULL, entity_count INTEGER NOT NULL, chunk_index INTEGER NOT NULL, embedding {vector_column})""",
                """CREATE TABLE IF NOT EXISTS comparisons (id BIGSERIAL PRIMARY KEY, query TEXT NOT NULL, timestamp TEXT NOT NULL, method TEXT NOT NULL, latency DOUBLE PRECISION NOT NULL, tokens_used INTEGER NOT NULL, chunks_retrieved INTEGER NOT NULL, relevance_score DOUBLE PRECISION NOT NULL, confidence_score DOUBLE PRECISION NOT NULL, answer TEXT NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS usage (id BIGSERIAL PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, input_tokens BIGINT NOT NULL, output_tokens BIGINT NOT NULL, cost_usd DOUBLE PRECISION NOT NULL, created_at TEXT NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS credit_accounts (provider TEXT PRIMARY KEY, initial_usd DOUBLE PRECISION NOT NULL, initial_tokens BIGINT NOT NULL, created_at TEXT NOT NULL)""",
                "CREATE INDEX IF NOT EXISTS idx_chunks_partition ON chunks(partition_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)",
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage(provider)",
            ]
            for statement in statements:
                self._query(statement)
            self._ensure_legacy_columns(vector_column)
            if self.pgvector_enabled:
                try:
                    self._query("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
                except Exception as exc:
                    print(f"pgvector HNSW index not created: {exc}")
            # Disk-backed lexical search: a generated tsvector column + GIN
            # index. Ranking happens in Postgres via ts_rank_cd, so a query
            # only pulls back matching rows -- the corpus never needs to be
            # loaded into Python to be searched.
            try:
                self._query("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_text_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED")
                self._query("CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (chunk_text_tsv)")
            except Exception as exc:
                print(f"Postgres full-text index not created (falling back to LIKE-based search): {exc}")
            self._query("""CREATE TABLE IF NOT EXISTS chunk_entities (chunk_id TEXT NOT NULL, entity TEXT NOT NULL, partition_id TEXT NOT NULL, PRIMARY KEY (chunk_id, entity))""")
            self._query("CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity)")
            self._query("CREATE INDEX IF NOT EXISTS idx_chunk_entities_partition ON chunk_entities(partition_id)")
            self._query("""CREATE TABLE IF NOT EXISTS entity_frequency (entity TEXT PRIMARY KEY, freq INTEGER NOT NULL)""")
        else:
            self._query("""CREATE TABLE IF NOT EXISTS documents (file_path TEXT PRIMARY KEY, file_name TEXT NOT NULL, partition_id TEXT NOT NULL, processed_date TEXT NOT NULL, chunk_count INTEGER NOT NULL, content_hash TEXT)""")
            self._query("""CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, partition_id TEXT NOT NULL, chunk_text TEXT NOT NULL, token_count INTEGER NOT NULL, entity_count INTEGER NOT NULL, chunk_index INTEGER NOT NULL, embedding TEXT)""")
            self._query("""CREATE TABLE IF NOT EXISTS comparisons (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, timestamp TEXT NOT NULL, method TEXT NOT NULL, latency REAL NOT NULL, tokens_used INTEGER NOT NULL, chunks_retrieved INTEGER NOT NULL, relevance_score REAL NOT NULL, confidence_score REAL NOT NULL, answer TEXT NOT NULL)""")
            self._query("""CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL, created_at TEXT NOT NULL)""")
            self._query("""CREATE TABLE IF NOT EXISTS credit_accounts (provider TEXT PRIMARY KEY, initial_usd REAL NOT NULL, initial_tokens INTEGER NOT NULL, created_at TEXT NOT NULL)""")
            for index in (
                "CREATE INDEX IF NOT EXISTS idx_chunks_partition ON chunks(partition_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)",
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage(provider)",
            ):
                self._query(index)
            self._ensure_legacy_columns("TEXT")
            # Disk-backed lexical search: FTS5, so a query only touches the
            # rows it matches instead of requiring every chunk to be
            # tokenized and held in RAM. 'porter' stemming keeps this close
            # in spirit to the existing stem_token()-based TF-IDF path.
            self._query(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                "chunk_id UNINDEXED, partition_id UNINDEXED, chunk_text, "
                "tokenize='porter unicode61')"
            )
            # Live per-term document-frequency view over chunks_fts's own
            # index -- no backfill needed (unlike entity_frequency), since
            # this reads directly from the FTS5 index structure. Used by
            # fts_search to cap how many high-frequency terms go into the
            # OR-joined MATCH query (see MAX_FTS_TERM_FANOUT below): a long
            # question can generate 15-20+ query terms, and BM25 already
            # discounts common terms in scoring, but FTS5 still has to
            # evaluate every chunk containing ANY of them before it can
            # rank and apply LIMIT -- confirmed via profiling to dominate
            # retrieval time on long, cross-domain questions.
            try:
                self._query("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_vocab USING fts5vocab(chunks_fts, 'row')")
            except Exception as exc:
                print(f"fts5vocab not available ({exc}); fts_search will run without term-frequency capping.")
            # Disk-backed entity graph: one row per (chunk, entity), instead
            # of the in-RAM entity_chunks/neighbours sets. partition_id is
            # denormalised onto each row so cross-domain partition detection
            # never needs a join against a full chunk-metadata table.
            self._query(
                "CREATE TABLE IF NOT EXISTS chunk_entities (chunk_id TEXT NOT NULL, "
                "entity TEXT NOT NULL, partition_id TEXT NOT NULL, "
                "PRIMARY KEY (chunk_id, entity))"
            )
            self._query("CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity)")
            self._query("CREATE INDEX IF NOT EXISTS idx_chunk_entities_partition ON chunk_entities(partition_id)")
            # Precomputed per-entity corpus-wide frequency, used by
            # entity_search/get_partition_ids_for_entities to cheaply filter
            # out high-fanout entities before the expensive aggregation --
            # see _filter_high_fanout_entities. Small table (one row per
            # unique entity, not per chunk_entities row), populated by
            # backfill_entity_frequency.py.
            self._query("""CREATE TABLE IF NOT EXISTS entity_frequency (entity TEXT PRIMARY KEY, freq INTEGER NOT NULL)""")


    def _ensure_legacy_columns(self, vector_type: str) -> None:
        if self.use_postgres:
            rows = self._query("SELECT column_name FROM information_schema.columns WHERE table_name = ?", ("chunks",), fetch=True)
            columns = {row["column_name"] for row in rows}
            if "embedding" not in columns:
                self._query(f"ALTER TABLE chunks ADD COLUMN embedding {vector_type}")
        else:
            rows = self._query("PRAGMA table_info(chunks)", fetch=True)
            columns = {row["name"] for row in rows}
            if "embedding" not in columns:
                self._query("ALTER TABLE chunks ADD COLUMN embedding TEXT")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM documents ORDER BY processed_date DESC", fetch=True)

    def get_all_chunks(self) -> Tuple[Dict[str, str], Dict[str, ChunkMetadata]]:
        rows = self._query("SELECT * FROM chunks ORDER BY file_path, chunk_index", fetch=True)
        texts: Dict[str, str] = {}
        metadata: Dict[str, ChunkMetadata] = {}
        for row in rows:
            texts[row["chunk_id"]] = clean_text(row["chunk_text"])
            metadata[row["chunk_id"]] = ChunkMetadata(row["chunk_id"], row["file_path"], row["partition_id"], int(row["chunk_index"]), int(row["token_count"]), int(row["entity_count"]))
        return texts, metadata

    def get_content_hashes(self) -> Dict[str, str]:
        rows = self._query("SELECT content_hash, file_path FROM documents WHERE content_hash IS NOT NULL", fetch=True)
        return {row["content_hash"]: row["file_path"] for row in rows}

    def save_document(self, file_path: str, partition_id: str, chunks: Dict[str, str], metadata: Dict[str, ChunkMetadata], content_hash: str = "") -> None:
        canonical_path = str(Path(file_path).resolve())
        conn = self._connect(); conn.execute("PRAGMA journal_mode=WAL")
        try:
            cursor = conn.cursor()
            marker = "%s" if self.use_postgres else "?"
            cursor.execute(f"DELETE FROM chunks WHERE file_path = {marker}", (canonical_path,))
            if self.use_postgres:
                cursor.execute("""INSERT INTO documents (file_path, file_name, partition_id, processed_date, chunk_count, content_hash) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (file_path) DO UPDATE SET file_name=EXCLUDED.file_name, partition_id=EXCLUDED.partition_id, processed_date=EXCLUDED.processed_date, chunk_count=EXCLUDED.chunk_count, content_hash=EXCLUDED.content_hash""", (canonical_path, Path(canonical_path).name, partition_id, datetime.now().isoformat(timespec="seconds"), len(chunks), content_hash))
            else:
                cursor.execute("""INSERT OR REPLACE INTO documents (file_path, file_name, partition_id, processed_date, chunk_count, content_hash) VALUES (?, ?, ?, ?, ?, ?)""", (canonical_path, Path(canonical_path).name, partition_id, datetime.now().isoformat(timespec="seconds"), len(chunks), content_hash))
            statement = "INSERT INTO chunks (chunk_id, file_path, partition_id, chunk_text, token_count, entity_count, chunk_index) VALUES (%s, %s, %s, %s, %s, %s, %s)" if self.use_postgres else "INSERT INTO chunks (chunk_id, file_path, partition_id, chunk_text, token_count, entity_count, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)"
            cursor.executemany(statement, [(cid, canonical_path, partition_id, text, metadata[cid].token_count, metadata[cid].entity_count, metadata[cid].chunk_index) for cid, text in chunks.items()])

            if not self.use_postgres:
                # SQLite only: mirror into the FTS5 table. (Postgres gets its
                # full-text index for free via the generated tsvector column
                # on `chunks` itself, so nothing extra to insert there.)
                cursor.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    [(cid,) for cid in chunks.keys()],
                )
                cursor.executemany(
                    "INSERT INTO chunks_fts (chunk_id, partition_id, chunk_text) VALUES (?, ?, ?)",
                    [(cid, partition_id, text) for cid, text in chunks.items()],
                )

            # Entity index (both backends): extract once, at ingest, and
            # store as plain rows -- this is what lets cross-domain routing
            # and graph-style ranking work without ever holding an
            # entity_chunks/neighbours structure in RAM.
            entity_rows = []
            for cid, text in chunks.items():
                for entity in extract_entities(text):
                    entity_rows.append((cid, entity, partition_id))
            del_stmt = "DELETE FROM chunk_entities WHERE chunk_id = %s" if self.use_postgres else "DELETE FROM chunk_entities WHERE chunk_id = ?"
            cursor.executemany(del_stmt, [(cid,) for cid in chunks.keys()])
            if entity_rows:
                ins_stmt = (
                    "INSERT INTO chunk_entities (chunk_id, entity, partition_id) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING"
                    if self.use_postgres else
                    "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity, partition_id) VALUES (?, ?, ?)"
                )
                cursor.executemany(ins_stmt, entity_rows)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Disk-backed search (used by ComparisonEngine in "lazy" / large-corpus
    # mode instead of holding document_tokens/entity_graph in RAM)
    # -------------------------------------------------------------------------
    MAX_FTS_TERM_FANOUT = int(os.getenv("MAX_FTS_TERM_FANOUT", "20000"))
    MAX_FTS_QUERY_TERMS = int(os.getenv("MAX_FTS_QUERY_TERMS", "6"))

    def _cap_fts_terms(self, clean_terms: List[str]) -> List[str]:
        """Keep only the rarest MAX_FTS_QUERY_TERMS terms (by corpus-wide
        document frequency, via the live fts5vocab view), dropping any
        term matching more than MAX_FTS_TERM_FANOUT chunks outright. A
        long question can generate 15-20+ terms after stopword filtering;
        BM25 already discounts common terms in the final ranking, so
        dropping the highest-frequency ones costs little accuracy while
        removing most of the evaluation cost. Falls through to the
        original (uncapped) term list if fts5vocab isn't available or the
        lookup fails, rather than breaking search.
        """
        if len(clean_terms) <= self.MAX_FTS_QUERY_TERMS:
            return clean_terms
        try:
            placeholders = ",".join("?" for _ in clean_terms)
            rows = self._query(
                f"SELECT term, doc FROM chunks_fts_vocab WHERE term IN ({placeholders})",
                clean_terms, fetch=True,
            )
            freq_map = {row["term"]: row["doc"] for row in rows}
            # Terms not found in the vocab (rare/misspelled/not yet indexed)
            # are treated as maximally rare -- freq 0 -- so they're kept.
            ranked_terms = sorted(clean_terms, key=lambda t: freq_map.get(t, 0))
            kept = [t for t in ranked_terms if freq_map.get(t, 0) <= self.MAX_FTS_TERM_FANOUT]
            return (kept or ranked_terms)[: self.MAX_FTS_QUERY_TERMS]
        except Exception:
            return clean_terms  # chunks_fts_vocab unavailable -- unfiltered behavior

    def fts_search(self, terms: Sequence[str], partition_id: Optional[str] = None, limit: int = 50) -> List[Tuple[str, float]]:
        """Lexical search that never loads the corpus into Python -- the
        database does the matching and ranking, and only `limit` results
        (chunk_id, relevance) come back. Higher score = more relevant,
        matching the convention every other ranking method in this file
        uses (unlike FTS5's raw bm25(), which is lower-is-better).
        """
        clean_terms = [t.replace('"', ' ').strip() for t in terms if t and t.strip()]
        if not clean_terms:
            return []
        if not self.use_postgres:
            clean_terms = self._cap_fts_terms(clean_terms)
        if self.use_postgres:
            tsquery = " | ".join(clean_terms)  # OR semantics, matches vector_rank's "any query term" behaviour
            sql = "SELECT chunk_id, ts_rank_cd(chunk_text_tsv, query) AS score FROM chunks, plainto_tsquery('english', %s) query WHERE chunk_text_tsv @@ query"
            params: List[Any] = [tsquery]
            if partition_id:
                sql += " AND partition_id = %s"
                params.append(partition_id)
            sql += " ORDER BY score DESC LIMIT %s"
            params.append(limit)
            rows = self._query(sql, params, fetch=True)
            return [(row["chunk_id"], float(row["score"])) for row in rows]
        else:
            match_query = " OR ".join(f'"{t}"' for t in clean_terms)
            sql = "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ?"
            params = [match_query]
            if partition_id:
                sql += " AND partition_id = ?"
                params.append(partition_id)
            sql += " ORDER BY score ASC LIMIT ?"  # FTS5 bm25() is a cost: lower = better match
            params.append(limit)
            rows = self._query(sql, params, fetch=True)
            # Invert sign so "higher = more relevant", consistent with the
            # rest of the ranking pipeline (RRF fusion, calibration, etc).
            return [(row["chunk_id"], -float(row["score"])) for row in rows]

    MAX_ENTITY_FANOUT = int(os.getenv("MAX_ENTITY_FANOUT", "5000"))

    def _filter_high_fanout_entities(self, entities_lower: List[str]) -> List[str]:
        """Drop entities matching more than MAX_ENTITY_FANOUT chunks
        corpus-wide, via a cheap lookup against the small precomputed
        entity_frequency table (see backfill_entity_frequency.py). Falls
        through to the original list, unfiltered, if that table doesn't
        exist yet or if every given entity happens to be high-frequency
        (searching something over searching nothing).
        """
        try:
            placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in entities_lower)
            freq_rows = self._query(
                f"SELECT entity, freq FROM entity_frequency WHERE entity IN ({placeholders})",
                entities_lower, fetch=True,
            )
            freq_map = {row["entity"]: row["freq"] for row in freq_rows}
            filtered = [e for e in entities_lower if freq_map.get(e, 0) <= self.MAX_ENTITY_FANOUT]
            return filtered if filtered else entities_lower
        except Exception:
            return entities_lower  # entity_frequency table doesn't exist yet

    def entity_search(self, entities: Iterable[str], partition_id: Optional[str] = None, limit: int = 100) -> List[Tuple[str, float]]:
        """Graph-style ranking without an in-RAM entity_chunks/neighbours
        structure: score a chunk by how many of the query's entities it
        also contains, via a single indexed SQL query.

        Entities are first filtered against entity_frequency (a small,
        precomputed table -- see backfill_entity_frequency.py) to drop
        anything matching more than MAX_ENTITY_FANOUT chunks corpus-wide.
        This is the general form of the "api"/"what"/"2026" problem found
        via profiling: a handful of specific words are now excluded at
        the stopword level, but any OTHER unexpectedly common term would
        cause the same expensive GROUP BY over hundreds of thousands of
        rows for near-zero discriminative benefit. Checking frequency via
        a primary-key lookup against a small table is cheap regardless of
        how large chunk_entities itself is; if entity_frequency hasn't
        been backfilled yet, this filter silently no-ops (falls through
        to the original, unfiltered behavior) rather than breaking search.
        """
        entities = [e for e in entities if e]
        if not entities:
            return []
        entities_lower = self._filter_high_fanout_entities([e.lower() for e in entities])

        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in entities_lower)
        sql = f"SELECT chunk_id, COUNT(DISTINCT entity) AS matches FROM chunk_entities WHERE entity IN ({placeholders})"
        params: List[Any] = list(entities_lower)
        if partition_id:
            sql += (" AND partition_id = %s" if self.use_postgres else " AND partition_id = ?")
            params.append(partition_id)
        sql += " GROUP BY chunk_id ORDER BY matches DESC"
        sql += (" LIMIT %s" if self.use_postgres else " LIMIT ?")
        params.append(limit)
        rows = self._query(sql, params, fetch=True)
        max_matches = max((row["matches"] for row in rows), default=1) or 1
        return [(row["chunk_id"], row["matches"] / max_matches) for row in rows]

    def get_partition_ids_for_entities(self, entities: Iterable[str]) -> List[str]:
        """Rank partitions by entity DENSITY, not raw hit count.

        Density = hits / partition_size. This stops big partitions (Slack
        800K chunks) from drowning out small relevant ones (Jira 25K)
        just because they have more total mentions of a common term.
        """
        entities = [e.lower() for e in entities if e]
        if not entities:
            return []
        entities = self._filter_high_fanout_entities(entities)
        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in entities)
        sql = (
            f"SELECT ce.partition_id, "
            f"COUNT(*) AS hits, "
            f"pc.total_chunks, "
            f"1.0 * COUNT(*) / pc.total_chunks AS density "
            f"FROM chunk_entities ce "
            f"JOIN (SELECT partition_id, COUNT(*) AS total_chunks FROM chunks GROUP BY partition_id) pc "
            f"ON ce.partition_id = pc.partition_id "
            f"WHERE ce.entity IN ({placeholders}) "
            f"GROUP BY ce.partition_id, pc.total_chunks "
            f"ORDER BY density DESC LIMIT 50"
        )
        rows = self._query(sql, entities, fetch=True)
        return [row["partition_id"] for row in rows]


    def get_chunk_texts(self, chunk_ids: Sequence[str]) -> Dict[str, str]:
        """Fetch text for a small, specific set of chunks (a router's
        candidate list, never the whole corpus)."""
        chunk_ids = list({cid for cid in chunk_ids if cid})
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in chunk_ids)
        rows = self._query(f"SELECT chunk_id, chunk_text FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids, fetch=True)
        return {row["chunk_id"]: clean_text(row["chunk_text"]) for row in rows}

    def get_chunk_metadata_batch(self, chunk_ids: Sequence[str]) -> Dict[str, ChunkMetadata]:
        """Fetch metadata for a small, specific set of chunks."""
        chunk_ids = list({cid for cid in chunk_ids if cid})
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in chunk_ids)
        rows = self._query(
            f"SELECT chunk_id, file_path, partition_id, chunk_index, token_count, entity_count "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids, fetch=True,
        )
        return {
            row["chunk_id"]: ChunkMetadata(
                row["chunk_id"], row["file_path"], row["partition_id"],
                int(row["chunk_index"]), int(row["token_count"]), int(row["entity_count"]),
            )
            for row in rows
        }

    def get_partition_counts(self) -> Dict[str, int]:
        """Cheap partition sizes (a GROUP BY COUNT, not a full row fetch) --
        used for router/context stats without materialising chunk-id lists."""
        rows = self._query("SELECT partition_id, COUNT(*) AS n FROM chunks GROUP BY partition_id", fetch=True)
        return {row["partition_id"]: int(row["n"]) for row in rows}

    def save_embeddings(self, embeddings: Dict[str, Any]) -> None:
        if not embeddings:
            return
        records = []
        for chunk_id, vector in embeddings.items():
            values = vector.detach().cpu().tolist() if hasattr(vector, "detach") else list(vector)
            records.append((json.dumps(values), chunk_id))
        if self.use_postgres and self.pgvector_enabled:
            self._many("UPDATE chunks SET embedding = ?::vector WHERE chunk_id = ?", records)
        else:
            self._many("UPDATE chunks SET embedding = ? WHERE chunk_id = ?", records)

    def save_comparison(self, query: str, results: Dict[str, Dict[str, Any]]) -> None:
        records = [(query, datetime.now().isoformat(timespec="seconds"), method, float(data.get("latency", 0)), int(data.get("tokens_used", 0)), int(data.get("chunks_retrieved", 0)), float(data.get("relevance_score", 0)), float(data.get("confidence_score", 0)), str(data.get("answer", ""))) for method, data in results.items()]
        self._many("INSERT INTO comparisons (query, timestamp, method, latency, tokens_used, chunks_retrieved, relevance_score, confidence_score, answer) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", records)

    @staticmethod
    def _credit_prefix(provider: str) -> str:
        return {"claude": "CLAUDE", "openai": "OPENAI", "gemini": "GEMINI"}[provider]

    def get_credit_status(self, provider: str) -> Dict[str, Any]:
        prefix = self._credit_prefix(provider)
        usd_raw, token_raw = os.getenv(f"{prefix}_CREDITS_USD"), os.getenv(f"{prefix}_CREDITS_TOKENS")
        configured_in_env = usd_raw not in (None, "") or token_raw not in (None, "")
        account_rows = self._query("SELECT initial_usd, initial_tokens FROM credit_accounts WHERE provider = ?", (provider,), fetch=True)
        if not account_rows and configured_in_env:
            self._query("INSERT INTO credit_accounts (provider, initial_usd, initial_tokens, created_at) VALUES (?, ?, ?, ?)", (provider, float(usd_raw or 0.0), int(token_raw or 0), datetime.now().isoformat(timespec="seconds")))
            account_rows = self._query("SELECT initial_usd, initial_tokens FROM credit_accounts WHERE provider = ?", (provider,), fetch=True)
        configured = bool(account_rows)
        initial_usd = float(account_rows[0]["initial_usd"]) if account_rows else 0.0
        initial_tokens = int(account_rows[0]["initial_tokens"]) if account_rows else 0
        rows = self._query("SELECT COALESCE(SUM(input_tokens), 0) AS input_tokens, COALESCE(SUM(output_tokens), 0) AS output_tokens, COALESCE(SUM(cost_usd), 0) AS used_usd FROM usage WHERE provider = ?", (provider,), fetch=True)
        used = rows[0]
        used_tokens = int(used["input_tokens"] or 0) + int(used["output_tokens"] or 0)
        return {"provider": provider, "configured": configured, "initial_usd": initial_usd, "initial_tokens": initial_tokens, "used_usd": float(used["used_usd"] or 0), "used_tokens": used_tokens, "remaining_usd": max(0.0, initial_usd - float(used["used_usd"] or 0)), "remaining_tokens": max(0, initial_tokens - used_tokens)}

    def get_all_credit_statuses(self) -> List[Dict[str, Any]]:
        return [self.get_credit_status(provider) for provider in self.CREDIT_PROVIDERS]

    def can_generate(self, provider: str, input_tokens: int, max_output_tokens: int) -> Tuple[bool, str]:
        status = self.get_credit_status(provider)
        if not status["configured"]:
            return True, "Credit limit is not configured."
        estimated_cost = estimate_provider_cost(provider, input_tokens, max_output_tokens)
        if status["remaining_usd"] < estimated_cost:
            return False, f"{provider.title()} credits are exhausted or insufficient (${status['remaining_usd']:.4f} remaining)."
        if status["remaining_tokens"] < input_tokens + max_output_tokens:
            return False, f"{provider.title()} token credits are exhausted or insufficient ({status['remaining_tokens']:,} remaining)."
        return True, ""

    def record_usage(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = estimate_provider_cost(provider, input_tokens, output_tokens)
        self._query("INSERT INTO usage (provider, model, input_tokens, output_tokens, cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?)", (provider, model, input_tokens, output_tokens, cost, datetime.now().isoformat(timespec="seconds")))
        return cost


def estimate_provider_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    prefix = {"claude": "CLAUDE", "openai": "OPENAI", "gemini": "GEMINI"}.get(provider, "OPENAI")
    default_input, default_output = PROVIDER_PRICING_DEFAULTS.get(provider, PROVIDER_PRICING_DEFAULTS["openai"])
    input_price = float(os.getenv(f"{prefix}_INPUT_PRICE", str(default_input)))
    output_price = float(os.getenv(f"{prefix}_OUTPUT_PRICE", str(default_output)))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@dataclass
class IngestionSummary:
    processed: int = 0
    duplicates: List[str] = None
    skipped: List[str] = None
    failures: List[str] = None
    partitions: Dict[str, int] = None

    def __post_init__(self) -> None:
        self.duplicates = self.duplicates or []
        self.skipped = self.skipped or []
        self.failures = self.failures or []
        self.partitions = self.partitions or {}

    def as_markdown(self) -> str:
        lines = [f"**Ingestion complete:** {self.processed} file(s) indexed."]
        if self.partitions:
            lines.append("Partitions: " + ", ".join(f"`{name.replace('partition_', '')}`: {count}" for name, count in sorted(self.partitions.items())))
        if self.duplicates:
            lines.append("**Duplicates ignored:** " + ", ".join(self.duplicates))
        if self.skipped:
            lines.append("**Unsupported/skipped:** " + ", ".join(self.skipped))
        if self.failures:
            lines.append("**Could not ingest:** " + "; ".join(self.failures))
        return "\n\n".join(lines)


def _hash_file_streaming(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a file's contents without reading it into memory at
    once -- matters once files (or the zips containing them) get into
    the hundreds of MB / low GB range."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


class DocumentProcessor:
    def __init__(self, db: DatabaseManager, lazy: bool = False):
        self.db = db
        self.lazy = lazy
        if lazy:
            # Large-corpus mode: don't load every chunk's text/metadata into
            # RAM up front. ComparisonEngine queries the database directly
            # (FTS5 + chunk_entities) instead of these dicts when lazy=True;
            # they're kept as empty dicts rather than removed so any code
            # that still reads processor.chunk_texts as a dict doesn't crash,
            # it just correctly sees nothing resident (use db methods for
            # actual lookups instead).
            self.chunk_texts, self.chunks_metadata = {}, {}
        else:
            self.chunk_texts, self.chunks_metadata = self.db.get_all_chunks()

    @staticmethod
    def _load_document(file_path: str) -> str:
        path = Path(file_path.strip().strip('"\''))
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                import fitz
            except ImportError as exc:
                raise ImportError("PDF support requires: pip install pymupdf") from exc
            with fitz.open(path) as pdf:
                return clean_text("\n".join(page.get_text() for page in pdf))
        if suffix in {".txt", ".md", ".py", ".js"}:
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if suffix == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                cells = notebook.get("cells", [])
                return clean_text("\n\n".join("".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", "")) for cell in cells if cell.get("cell_type") in {"markdown", "code"}))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid notebook JSON: {path.name}") from exc
        raise ValueError(f"Unsupported file type: {suffix}")

    _CITATION_MARKER_RE = re.compile(r"\[\d{1,3}(?:,\s*\d{1,3})*\]|\(\d{4}[a-z]?\)|et al\.")

    @classmethod
    def _is_non_content_chunk(cls, text: str) -> bool:
        opening = text[:180].lower().strip()
        if (
            len(text.split()) < 25
            or opening.startswith("references")
            or opening.startswith("bibliography")
            or opening.startswith("acknowledg")
        ):
            return True
        words = text.split()
        if not words:
            return True
        markers = cls._CITATION_MARKER_RE.findall(text)
        citation_density_per_100_words = (len(markers) / max(1, len(words))) * 100
        if citation_density_per_100_words >= 12.0:
            return True
        return False

    def _classify_document(self, text: str, file_path: str) -> str:
        """
        Assigns a partition ID based on the immediate parent folder of the
        file. If the file is at the root, or its folder name carries no
        signal (e.g. "documents", "uploads"), partition = 'general'.
        """
        path = Path(file_path)
        parent = path.parent.name
        if parent and parent != "." and parent.lower() not in GENERIC_FOLDER_NAMES:
            return f"partition_{parent}"
        return "partition_general"

    def _keyword_label(self, text: str, top_n: int = 2) -> str:
        """Cheap, corpus-agnostic label for a single document: its own
        most frequent non-trivial words. Needs only the document's own
        text (no corpus-wide stats), so it's safe to use per-file even
        for very large corpora where clustering the whole corpus at once
        isn't practical.
        """
        window = text[:4000].lower()
        tokens = [
            stem_token(token) for token in re.findall(r"\b[a-z][a-z-]{3,}\b", window)
            if token not in STOPWORDS
        ]
        if not tokens:
            return "general"
        top = [term for term, _ in Counter(tokens).most_common(top_n)]
        return "_".join(top) or "general"

    # --------------------------------------------------------------------
    # NEW: Auto‑partitioning via content clustering (dump‑and‑go)
    # --------------------------------------------------------------------
    def _auto_partition_documents(self, file_paths: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Automatically assign partition IDs for a flat, un-foldered corpus.
        Returns (partition_map, preloaded_texts) -- texts are handed back
        so process_uploads() can pass them straight into ingestion instead
        of re-parsing every PDF/notebook a second time, which matters once
        a corpus gets into the hundreds of MB.

        Falls back to a cheap per-document keyword label (no clustering,
        no corpus held in memory at once) when scikit-learn isn't
        installed, or when the corpus is too large to cluster cheaply --
        this is what keeps a ~1.5GB upload fast instead of stalling on a
        single giant TF-IDF/KMeans pass.
        """
        texts: Dict[str, str] = {}
        total_chars = 0
        for fp in file_paths:
            try:
                text = self._load_document(fp)
            except Exception:
                continue
            if text and len(text.split()) > 50:  # ignore very short docs
                texts[fp] = text
                total_chars += len(text)

        if not texts:
            return {fp: "partition_general" for fp in file_paths}, {}

        def _keyword_fallback() -> Dict[str, str]:
            mapping = {fp: f"partition_{self._keyword_label(text)}" for fp, text in texts.items()}
            for fp in file_paths:
                mapping.setdefault(fp, "partition_general")
            return mapping

        if total_chars > AUTO_PARTITION_MAX_CORPUS_CHARS:
            print(
                f"Corpus text ({total_chars:,} chars) exceeds AUTO_PARTITION_MAX_CORPUS_CHARS "
                f"({AUTO_PARTITION_MAX_CORPUS_CHARS:,}); using per-document keyword "
                "partitioning instead of whole-corpus clustering."
            )
            return _keyword_fallback(), texts

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.decomposition import TruncatedSVD
            import numpy as np
        except ImportError:
            print(
                "scikit-learn not installed; using per-document keyword partitioning "
                "(pip install scikit-learn for content-clustered partitions)."
            )
            return _keyword_fallback(), texts

        valid_paths = list(texts.keys())
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = vectorizer.fit_transform(texts.values())

        n_components = min(100, X.shape[1] - 1, len(valid_paths) - 1)
        if n_components < 2:
            partition_map = {fp: "partition_general" for fp in valid_paths}
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = svd.fit_transform(X)
            n_docs = len(valid_paths)
            n_clusters = min(max(2, int(np.sqrt(n_docs))), 10, n_docs)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_reduced)
            partition_map = {fp: f"partition_cluster_{label}" for fp, label in zip(valid_paths, labels)}
        for fp in file_paths:
            partition_map.setdefault(fp, "partition_general")
        return partition_map, texts

    def _chunk_document(
        self, text: str, source_path: str, partition_id: str
    ) -> Tuple[Dict[str, str], Dict[str, ChunkMetadata]]:
        words = clean_text(text).split()
        chunks: Dict[str, str] = {}
        metadata: Dict[str, ChunkMetadata] = {}
        source_hash = hashlib.sha1(str(Path(source_path).resolve()).encode("utf-8")).hexdigest()[:12]
        step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
        chunk_number = 0
        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_WORDS]
            if len(chunk_words) < 25:
                continue
            chunk_text = " ".join(chunk_words)
            if self._is_non_content_chunk(chunk_text):
                continue
            chunk_number += 1
            chunk_id = f"{source_hash}_{chunk_number:04d}"
            entities = extract_entities(chunk_text)
            chunks[chunk_id] = chunk_text
            metadata[chunk_id] = ChunkMetadata(
                chunk_id=chunk_id,
                file_path=str(Path(source_path).resolve()),
                partition_id=partition_id,
                chunk_index=chunk_number,
                token_count=estimate_tokens(chunk_text),
                entity_count=len(entities),
            )
            if start + CHUNK_WORDS >= len(words):
                break
        return chunks, metadata

    def _extract_zip(self, archive_path: Path, summary: IngestionSummary) -> Tuple[List[Path], Set[str]]:
        """Extract supported members, preserving the archive's internal
        folder structure on disk (so hr/policy.md stays under an "hr"
        folder after extraction -- this is what lets _classify_document's
        folder-based partitioning pick it up unchanged).

        Also returns the set of extracted paths that sat directly at the
        archive's root with no internal subfolder. Those files' on-disk
        parent will just be this archive's own randomly-named extraction
        destination -- a real signal for nothing -- so callers must not
        mistake it for a curated category folder the way a genuine
        internal subfolder (hr/, legal/, ...) is.
        """
        try:
            if not zipfile.is_zipfile(archive_path):
                summary.failures.append(f"{archive_path.name} (invalid ZIP archive)")
                return [], set()
            archive_hash = _hash_file_streaming(archive_path)[:16]
            destination = (UPLOAD_DIR / f"{archive_path.stem}_{archive_hash}").resolve()
            destination.mkdir(parents=True, exist_ok=True)
            extracted: List[Path] = []
            flat_root_paths: Set[str] = set()
            with zipfile.ZipFile(archive_path) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                if len(members) > MAX_ZIP_FILES:
                    summary.failures.append(f"{archive_path.name} (contains {len(members)} files; limit is {MAX_ZIP_FILES})")
                    return [], set()
                for member in members:
                    member_path = Path(member.filename)
                    suffix = member_path.suffix.lower()
                    if suffix not in SUPPORTED_EXTENSIONS:
                        summary.skipped.append(f"{member.filename} (unsupported {suffix or 'file type'})")
                        continue
                    if member.file_size > MAX_ZIP_MEMBER_BYTES:
                        summary.skipped.append(f"{member.filename} (larger than {MAX_ZIP_MEMBER_BYTES // (1024 * 1024)} MB limit)")
                        continue
                    target = (destination / member_path).resolve()
                    if destination not in target.parents:
                        summary.skipped.append(f"{member.filename} (unsafe archive path)")
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        output.write(source.read())
                    extracted.append(target)
                    if member_path.parent == Path("."):
                        flat_root_paths.add(str(target))
            return extracted, flat_root_paths
        except (OSError, zipfile.BadZipFile) as exc:
            summary.failures.append(f"{archive_path.name} ({exc})")
            return [], set()

    # --------------------------------------------------------------------
    # MODIFIED: _process_document now accepts partition_map
    # --------------------------------------------------------------------
    def _process_document(self, path: Path, existing_hashes: Dict[str, str], force: bool,
                          summary: IngestionSummary, partition_map: Dict[str, str] = None,
                          preloaded_text: Optional[str] = None) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            summary.skipped.append(f"{path.name} (unsupported {path.suffix or 'file type'})")
            return
        if not path.exists():
            summary.failures.append(f"{path.name} (file not found)")
            return
        canonical_path = str(path.resolve())
        try:
            content_hash = _hash_file_streaming(path)
            if not force and content_hash in existing_hashes:
                summary.duplicates.append(f"{path.name} (same as {Path(existing_hashes[content_hash]).name})")
                return
            text = preloaded_text if preloaded_text is not None else self._load_document(canonical_path)
            if not text:
                summary.skipped.append(f"{path.name} (empty)")
                return
            # Determine partition: if partition_map provided, use it; else fallback to folder‑based
            if partition_map and canonical_path in partition_map:
                partition = partition_map[canonical_path]
            else:
                partition = self._classify_document(text, canonical_path)
            chunks, metadata = self._chunk_document(text, canonical_path, partition)
            if not chunks:
                summary.skipped.append(f"{path.name} (no usable content chunks)")
                return
            if not self.lazy:
                old_chunk_ids = [chunk_id for chunk_id, old_metadata in self.chunks_metadata.items() if old_metadata.file_path == canonical_path]
                for chunk_id in old_chunk_ids:
                    self.chunk_texts.pop(chunk_id, None)
                    self.chunks_metadata.pop(chunk_id, None)
            self.db.save_document(canonical_path, partition, chunks, metadata, content_hash=content_hash)
            if not self.lazy:
                self.chunk_texts.update(chunks)
                self.chunks_metadata.update(metadata)
            existing_hashes[content_hash] = canonical_path
            summary.processed += 1
            summary.partitions[partition] = summary.partitions.get(partition, 0) + 1
        except Exception as exc:
            summary.failures.append(f"{path.name} ({exc})")

    # --------------------------------------------------------------------
    # MODIFIED: process_uploads with auto‑partitioning trigger
    # --------------------------------------------------------------------
    def process_uploads(self, file_paths: Sequence[str], force: bool = False) -> IngestionSummary:
        """Ingest direct files and ZIP archives, with auto-partitioning if needed.

        ZIPs are extracted first so the partitioning decision is made
        against the files that will actually be indexed -- including a
        ZIP's own internal folder structure (e.g. hr/, legal/, ...), which
        is used as-is when it exists. Content-based clustering only
        kicks in for genuinely flat corpora (no folder signal at all,
        whether uploaded directly or inside a ZIP), so it never
        overrides a folder structure someone already curated.
        """
        existing_hashes = self.db.get_content_hashes()
        summary = IngestionSummary()

        resolved_paths: List[Path] = []
        flat_root_paths: Set[str] = set()  # extracted members with no internal zip subfolder
        for raw_path in file_paths:
            path = Path(raw_path.strip().strip('"\''))
            if path.suffix.lower() == ".zip":
                extracted, flat_roots = self._extract_zip(path, summary)  # already resolved paths
                resolved_paths.extend(extracted)
                flat_root_paths.update(flat_roots)
            elif path.exists():
                resolved_paths.append(path.resolve())
            else:
                resolved_paths.append(path)  # let _process_document report "file not found"

        meaningful_parents = {
            p.parent.name for p in resolved_paths
            if str(p) not in flat_root_paths
            and p.parent.name
            and p.parent.name.lower() not in GENERIC_FOLDER_NAMES
        }
        use_auto_partition = len(meaningful_parents) <= 1

        partition_map: Dict[str, str] = {}
        preloaded_texts: Dict[str, str] = {}
        if use_auto_partition and resolved_paths:
            partition_map, preloaded_texts = self._auto_partition_documents(
                [str(p) for p in resolved_paths]
            )

        for path in resolved_paths:
            self._process_document(
                path, existing_hashes, force, summary, partition_map,
                preloaded_text=preloaded_texts.get(str(path)),
            )
        return summary

    def process_documents(self, file_paths: Sequence[str], force: bool = False) -> int:
        summary = self.process_uploads(file_paths, force=force)
        print(summary.as_markdown())
        return summary.processed


def select_files_dialog() -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Select research papers",
            filetypes=[("Supported research files", "*.pdf *.txt *.md *.py *.js *.ipynb *.zip"), ("All files", "*.*")],
        )
        root.destroy()
        return list(files)
    except Exception as exc:
        print(f"File picker unavailable: {exc}")
        return []


# ---------------------------------------------------------------------------
# Retrieval primitives
# ---------------------------------------------------------------------------
def build_tfidf_index(chunks: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    document_tokens = {
        chunk_id: [stem_token(token) for token in re.findall(r"\b[\w-]+\b", text.lower())]
        for chunk_id, text in chunks.items() if text
    }
    frequencies: Dict[str, int] = defaultdict(int)
    for tokens in document_tokens.values():
        for term in set(tokens):
            frequencies[term] += 1
    document_count = max(1, len(document_tokens))
    idf = {
        term: math.log((document_count + 1) / (frequency + 1)) + 1
        for term, frequency in frequencies.items()
    }
    return document_tokens, idf


def vector_rank(
    documents: Dict[str, List[str]], idf: Dict[str, float], terms: Sequence[str]
) -> List[Tuple[str, float]]:
    if not documents or not terms:
        return []
    query_tf = Counter(terms)
    query_vector = {
        term: count * idf.get(term, 1.0)
        for term, count in query_tf.items()
    }
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
    scores = []
    for chunk_id, tokens in documents.items():
        term_frequency = Counter(tokens)
        dot = sum(
            term_frequency.get(term, 0) * idf.get(term, 1.0) * query_weight
            for term, query_weight in query_vector.items()
        )
        document_norm = math.sqrt(sum(
            (count * idf.get(term, 1.0)) ** 2
            for term, count in term_frequency.items()
        )) or 1.0
        scores.append((chunk_id, dot / (query_norm * document_norm)))
    return sorted(scores, key=lambda item: item[1], reverse=True)


class EntityKnowledgeGraph:
    def __init__(self, chunks: Dict[str, str]):
        self.chunk_entities: Dict[str, Set[str]] = {}
        self.entity_chunks: Dict[str, Set[str]] = defaultdict(set)
        self.neighbours: Dict[str, Set[str]] = defaultdict(set)
        for chunk_id, text in chunks.items():
            entities = extract_entities(text)
            self.chunk_entities[chunk_id] = entities
            for entity in entities:
                self.entity_chunks[entity].add(chunk_id)
                self.neighbours[entity].update(entities - {entity})

    def rank(self, query: str, candidate_ids: Iterable[str]) -> List[Tuple[str, float]]:
        query_entities = extract_entities(query)
        if not query_entities:
            return []
        ranked = []
        for chunk_id in candidate_ids:
            entities = self.chunk_entities.get(chunk_id, set())
            direct = (len(entities & query_entities) / len(query_entities)) if query_entities else 0.0
            connected = (
                sum(
                    any(
                        entity in self.neighbours.get(query_entity, set())
                        for query_entity in query_entities
                    )
                    for entity in entities
                ) / max(1, len(entities))
                if query_entities else 0.0
            )
            score = (0.75 * direct) + (0.25 * connected)
            if score > 0:
                ranked.append((chunk_id, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------------------------
# Embedding-based semantic ranker (optional)
# ---------------------------------------------------------------------------
class EmbeddingRanker:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.model = None
        self.db = db
        self.chunk_embeddings: Dict[str, Any] = {}
        if _EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                print(f"✅ Loaded embedding model: {EMBEDDING_MODEL_NAME}")
            except Exception as e:
                print(f"⚠️ Failed to load embedding model: {e}")
                self.model = None

    def build_index(self, chunk_texts: Dict[str, str]) -> None:
        if self.model is None:
            return
        if not chunk_texts:
            return
        texts = list(chunk_texts.values())
        chunk_ids = list(chunk_texts.keys())
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            self.chunk_embeddings = {cid: emb for cid, emb in zip(chunk_ids, embeddings)}
            if self.db is not None:
                self.db.save_embeddings(self.chunk_embeddings)
        except Exception as e:
            print(f"⚠️ Embedding indexing failed: {e}")
            self.chunk_embeddings = {}

    def rank(self, query: str, candidate_ids: Iterable[str]) -> List[Tuple[str, float]]:
        if self.model is None or not self.chunk_embeddings:
            return []
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = []
        for cid in candidate_ids:
            if cid not in self.chunk_embeddings:
                continue
            sim = (query_emb @ self.chunk_embeddings[cid]).item()
            scores.append((cid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def rank_on_demand(self, query: str, chunk_texts: Dict[str, str]) -> List[Tuple[str, float]]:
        """Embed and rank just the given (small) set of texts, fresh, with
        no persistent index. Used by the tier-4 last-resort escalation in
        large-corpus mode, where building/holding a full-corpus embedding
        index isn't an option under the memory budget -- this way
        embeddings only ever exist transiently, for a handful of already
        pre-filtered candidates, and are garbage-collected right after.
        """
        if self.model is None or not chunk_texts:
            return []
        try:
            chunk_ids = list(chunk_texts.keys())
            texts = list(chunk_texts.values())
            query_emb = self.model.encode(query, convert_to_tensor=True)
            chunk_embs = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            scores = [(cid, (query_emb @ emb).item()) for cid, emb in zip(chunk_ids, chunk_embs)]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores
        except Exception as e:
            print(f"⚠️ On-demand embedding ranking failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Cross-encoder reranker (optional)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker:
    def __init__(self):
        self.model = None
        if _CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
                print(f"✅ Loaded cross-encoder reranker: {CROSS_ENCODER_MODEL_NAME}")
            except Exception as e:
                print(f"⚠️ Failed to load cross-encoder reranker: {e}")
                self.model = None

    def rerank(self, query: str, chunk_texts: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        if self.model is None or not chunk_texts:
            return []
        try:
            pairs = [(query, text) for _, text in chunk_texts]
            scores = self.model.predict(pairs)
            return [(cid, float(score)) for (cid, _), score in zip(chunk_texts, scores)]
        except Exception as e:
            print(f"⚠️ Cross-encoder reranking failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------
@dataclass
class ProviderConfig:
    name: str
    model: str
    api_key: str
    base_url: Optional[str] = None


def available_provider_configs(preferred: str = "auto", key_override: Optional[str] = None) -> List[ProviderConfig]:
    def _resolve_key(name: str, env_var: str) -> str:
        if key_override and key_override.strip() and preferred == name:
            return key_override.strip()
        return os.getenv(env_var, "")

    configured = {
        "claude": ProviderConfig("claude", os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL), _resolve_key("claude", "ANTHROPIC_API_KEY")),
        "gemini": ProviderConfig("gemini", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), _resolve_key("gemini", "GEMINI_API_KEY")),
        "openai": ProviderConfig("openai", os.getenv("OPENAI_MODEL", "gpt-5.6-luna"), _resolve_key("openai", "OPENAI_API_KEY")),
        "compatible": ProviderConfig("compatible", os.getenv("COMPATIBLE_MODEL", ""), _resolve_key("compatible", "COMPATIBLE_API_KEY"), os.getenv("COMPATIBLE_BASE_URL", "")),
    }
    priority = ["claude", "gemini", "openai", "compatible"]
    if preferred != "auto":
        priority = [preferred] + [name for name in priority if name != preferred]
    return [configured[name] for name in priority if name in configured and configured[name].api_key and configured[name].model]


def choose_provider() -> str:
    choices = {"1": "auto", "2": "claude", "3": "gemini", "4": "openai", "5": "compatible"}
    print("\nGeneration provider: 1) Auto  2) Claude  3) Gemini  4) OpenAI  5) Compatible")
    return choices.get(input("Select provider [1]: ").strip() or "1", "auto")


class ComparisonEngine:
    """Holds the (expensive, read-only-after-build) retrieval index: TF-IDF,
    entity graph, embedding index, and cross-encoder. These are safe to
    share across many concurrent requests/threads since queries only read
    them.
    """

    def __init__(self, processor: DocumentProcessor, db: DatabaseManager, provider: str = "auto", api_key: Optional[str] = None):
        self.processor = processor
        self.db = db
        self.lazy = processor.lazy
        if self.lazy:
            # Large-corpus mode: no document_tokens, no entity_graph, no
            # full-corpus embedding index -- retrieval reads chunks_fts /
            # chunk_entities directly (see _rank_candidates,
            # _get_active_partitions) so memory stays bounded by query
            # candidate-set size, not corpus size.
            self.document_tokens, self.idf, self.entity_graph = {}, {}, None
            self.partition_chunks = self.db.get_partition_counts()  # {partition_id: count}, not chunk-id lists
            self.embedding_ranker = EmbeddingRanker(db)
            # Deliberately do NOT call build_index() here: that would
            # embed the whole corpus into RAM, exactly what lazy mode
            # exists to avoid. Embeddings are computed on demand, only for
            # a query's small candidate set, by the tier-4 escalation path.
        else:
            self.document_tokens, self.idf, self.entity_graph = self._build_index_streaming(processor.chunk_texts)
            self.partition_chunks: Dict[str, List[str]] = defaultdict(list)
            for chunk_id, metadata in processor.chunks_metadata.items():
                self.partition_chunks[metadata.partition_id].append(chunk_id)
            self.embedding_ranker = EmbeddingRanker(db)
            if self.embedding_ranker.model is not None:
                self.embedding_ranker.build_index(processor.chunk_texts)
        self.provider = provider
        self.provider_configs = available_provider_configs(provider, key_override=api_key)
        self.cross_encoder = CrossEncoderReranker()
        self.CROSS_DOMAIN_THRESHOLD_RELIEF = float(os.getenv("CROSS_DOMAIN_THRESHOLD_RELIEF", "0.12"))
        # Load per-customer routing config if available
        self._load_routing_config()
        self.MIN_THRESHOLD = float(os.getenv("MIN_THRESHOLD", "0.20"))
        # Smart router thresholds (see _should_use_hybrid).
        self.ROUTER_STANDARD_CONFIDENCE_HIGH = float(os.getenv("ROUTER_STANDARD_CONFIDENCE_HIGH", "0.99"))
        self.ROUTER_STANDARD_CONFIDENCE_LOW = float(os.getenv("ROUTER_STANDARD_CONFIDENCE_LOW", "0.40"))
        self.ROUTER_HYBRID_SCORE_THRESHOLD = float(os.getenv("ROUTER_HYBRID_SCORE_THRESHOLD", "0.50"))
        # Tier-4 last-resort threshold: if the disk-backed FTS5+graph
        # Hybrid tier is *still* not confident after running, escalate to
        # cross-encoder/embedding reranking of just its own top candidates
        # (never the full corpus). Only meaningful in lazy mode -- in
        # non-lazy mode cross-encoder/embeddings already run as part of
        # every Hybrid call, unchanged from before.
        self.ROUTER_LAST_RESORT_THRESHOLD = float(os.getenv("ROUTER_LAST_RESORT_THRESHOLD", "0.55"))
        self.LAST_RESORT_CANDIDATE_POOL = int(os.getenv("LAST_RESORT_CANDIDATE_POOL", "20"))

    def _load_routing_config(self, config_path: str = "routing_config.yaml") -> None:
        """Load per-customer partition routing keywords from a YAML config file.

        If the file doesn't exist, the hardcoded defaults in
        _DEFAULT_PARTITION_KEYWORDS are used. This lets each deployment
        customize routing without editing code.
        """
        try:
            import yaml
        except ImportError:
            return  # PyYAML not installed — use defaults
        if not os.path.exists(config_path):
            return  # no config file — use defaults
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if config and "partitions" in config:
                loaded = {}
                for partition_id, settings in config["partitions"].items():
                    keywords = settings.get("keywords", [])
                    if keywords:
                        loaded[partition_id] = [kw.lower() for kw in keywords]
                if loaded:
                    self.PARTITION_KEYWORDS = loaded
                    print(f"  [routing] Loaded {len(loaded)} partition keyword mappings from {config_path}")
        except Exception as e:
            print(f"  [routing] Could not load {config_path}: {e}. Using defaults.")

    def _build_index_streaming(
        self, chunk_texts: Dict[str, str], batch_size: int = 50000
    ) -> Tuple[Dict[str, List[str]], Dict[str, float], "EntityKnowledgeGraph"]:
        """Build the TF-IDF token index and the entity knowledge graph in
        batches instead of constructing every intermediate structure over
        the whole corpus in one shot.

        build_tfidf_index() and EntityKnowledgeGraph() each build a full
        tokenized/entity-extracted copy of every chunk plus several large
        supporting dicts (document_tokens, per-term frequency counts,
        entity_chunks, neighbours) all at once. On a ~1.9M-chunk corpus
        those coexist in memory simultaneously during construction, which
        is what blows up. Processing chunk_texts in batches keeps peak
        memory bounded by batch_size: each batch's transient locals go
        out of scope (and get gc'd) before the next batch starts.

        Every token and entity string is also passed through sys.intern()
        before being stored. A word like "compliance" appearing 50,000
        times across the corpus would otherwise allocate 50,000 separate
        (equal but distinct) string objects; interning makes all of them
        point at the same one object, so memory scales with vocabulary
        size (tens of thousands of unique terms/entities) instead of
        total token count (hundreds of millions). This changes nothing
        about tokenization, TF-IDF math, or entity matching -- Python
        strings compare and hash by value either way, so
        document_tokens, idf, and every ranking/retrieval method that
        reads them behave identically to before.

        The final document_tokens/idf/entity_graph end up identical in
        *content* to what building everything at once would have
        produced -- this only changes how (and how cheaply) they're
        built, not what they contain.
        """
        import gc

        document_tokens: Dict[str, List[str]] = {}
        self._doc_freq: Dict[str, int] = defaultdict(int)

        entity_graph = EntityKnowledgeGraph.__new__(EntityKnowledgeGraph)
        entity_graph.chunk_entities = {}
        entity_graph.entity_chunks = defaultdict(set)
        entity_graph.neighbours = defaultdict(set)

        chunk_ids = list(chunk_texts.keys())
        total = len(chunk_ids)
        for start in range(0, total, batch_size):
            batch_ids = chunk_ids[start:start + batch_size]

            for chunk_id in batch_ids:
                text = chunk_texts.get(chunk_id)
                if not text:
                    continue

                # --- TF-IDF: tokenize this chunk, intern each token so
                # repeated words share one string object, fold presence
                # into the running doc-freq counter ---
                tokens = [
                    sys.intern(stem_token(token))
                    for token in re.findall(r"\b[\w-]+\b", text.lower())
                ]
                document_tokens[chunk_id] = tokens
                for term in set(tokens):
                    self._doc_freq[term] += 1

                # --- Entity graph: same per-chunk extraction/merge the
                # original EntityKnowledgeGraph.__init__ did, with
                # entities interned for the same reason as tokens ---
                entities = {sys.intern(entity) for entity in extract_entities(text)}
                entity_graph.chunk_entities[chunk_id] = entities
                for entity in entities:
                    entity_graph.entity_chunks[entity].add(chunk_id)
                    entity_graph.neighbours[entity].update(entities - {entity})

            print(f"  Indexed {min(start + batch_size, total):,}/{total:,} chunks...")
            gc.collect()

        document_count = max(1, len(document_tokens))
        idf = {
            term: math.log((document_count + 1) / (frequency + 1)) + 1
            for term, frequency in self._doc_freq.items()
        }
        return document_tokens, idf, entity_graph

    # -------------------------------------------------------------------------
    # Domain detection using entity graph
    # -------------------------------------------------------------------------
    MAX_ACTIVE_PARTITIONS = 3  # cap: don't trigger cross-domain for all 9

    # Default keyword -> partition mapping. Loaded from routing_config.yaml
    # if it exists (per-customer config), otherwise uses these defaults.
    # To customize for a new deployment: edit routing_config.yaml, not this code.
    _DEFAULT_PARTITION_KEYWORDS = {
        "partition_jira": ["jira", "ticket", "epic", "sprint", "backlog", "kanban"],
        "partition_github": ["github", "pr ", "pull request", "pull-request", "repo", "repository", "commit", "branch", "merge request", "merge-request"],
        "partition_slack": ["slack", "channel", "dm ", "direct message", "slack message"],
        "partition_gmail": ["gmail", "email", "inbox", "thread", "mail "],
        "partition_confluence": ["confluence", "wiki", "page", "documentation page"],
        "partition_hubspot": ["hubspot", "deal", "contact", "company", "pipeline", "crm"],
        "partition_linear": ["linear", "linear issue", "project "],
        "partition_fireflies": ["fireflies", "meeting", "transcript", "call recording", "firefly"],
        "partition_google_drive": ["google drive", "drive", "doc ", "document", "sheet", "spreadsheet"],
    }
    PARTITION_KEYWORDS = _DEFAULT_PARTITION_KEYWORDS  # may be overridden in __init__

    def _get_active_partitions(self, query: str) -> List[str]:
        """Return partition IDs relevant to this query.

        Two-tier routing:
        1. KEYWORD MATCH: scan the query for partition-specific keywords.
           If "jira" is mentioned, partition_jira is active. This is fast,
           deterministic, and correct for explicit-source questions.
        2. ENTITY DENSITY FALLBACK: if no keywords matched, fall back to
           entity-density routing for questions that don't name a source.
        """
        query_lower = query.lower()

        # --- Tier 1: keyword matching ---
        matched = []
        for partition_id, keywords in self.PARTITION_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    if partition_id not in matched:
                        matched.append(partition_id)
                    break  # one keyword match is enough for this partition

        if matched:
            return matched[:3]

        # --- Tier 2: entity density fallback ---
        query_entities = extract_entities(query)
        if not query_entities:
            return []
        if self.lazy:
            return self.db.get_partition_ids_for_entities(query_entities)[:3]
        partition_scores = defaultdict(int)
        for entity in query_entities:
            for chunk_id in self.entity_graph.entity_chunks.get(entity, []):
                metadata = self.processor.chunks_metadata.get(chunk_id)
                if metadata:
                    partition_scores[metadata.partition_id] += 1
        sorted_partitions = sorted(partition_scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_partitions][:3]


    def _is_cross_domain_question(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) < 2:
            return False
        # 2+ active partitions = cross-domain, no comparison keywords needed
        return True


    def _is_simple_single_domain(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) != 1:
            return False
        terms = query_terms(query)
        return len(terms) <= 6

    def _get_adaptive_threshold(self, query: str, is_cross_domain: Optional[bool] = None) -> float:
        lower = query.lower()
        complexity_markers = (
            "compare", "contrast", "difference", "versus", " vs ",
            "relationship", "how does", "why does", "analyze", "analyse",
            "evaluate", "synthesize", "critique", "assess",
        )
        complexity = sum(marker in lower for marker in complexity_markers) + len(self._get_active_partitions(query))
        if complexity <= 2:
            base = 0.35
        elif complexity <= 4:
            base = 0.45
        elif complexity <= 6:
            base = 0.55
        else:
            base = 0.60
        if is_cross_domain is None:
            is_cross_domain = self._is_cross_domain_question(query)
        if is_cross_domain:
            base = max(self.MIN_THRESHOLD, base - self.CROSS_DOMAIN_THRESHOLD_RELIEF)
        return base

    def _max_chunks_for_query(self, query: str, is_cross_domain: bool) -> int:
        if self._is_simple_single_domain(query):
            return 2
        if is_cross_domain:
            partitions = self._get_active_partitions(query)
            num_domains = max(1, len(partitions))
            return min(
                MAX_CROSS_DOMAIN_CHUNKS_CEILING,
                max(MAX_CROSS_DOMAIN_CHUNKS, num_domains * MIN_CHUNKS_PER_ACTIVE_PARTITION)
            )
        complexity = len(query_terms(query))
        if complexity <= 6:
            return 2
        elif complexity <= 10:
            return 3
        elif complexity <= 15:
            return 4
        else:
            return MAX_CHUNKS

    # -------------------------------------------------------------------------
    # Smart router: Standard vs Hybrid
    # -------------------------------------------------------------------------
    def _cheap_standard_confidence(self, query: str) -> float:
        """Fast, retrieval-only probe used by the router: ranks with
        TF-IDF/FTS5 alone, keeps at most 2 chunks, and returns the
        resulting confidence. No generation call, no graph/embeddings/
        cross-encoder -- this is meant to be cheap enough to run before
        deciding whether Hybrid's extra latency and cost are worth it.
        """
        candidate_ids = None if self.lazy else list(self.processor.chunk_texts)
        ranked = self._rank_candidates(query, candidate_ids, use_hybrid=False)
        # confidence_threshold=1.1 is unreachable, so _intelligent_retrieve
        # never "succeeds" early -- it just walks up to 2 chunks and hands
        # back whatever confidence that yields, which is all we want here.
        _, confidence = self._intelligent_retrieve(
            ranked, query, confidence_threshold=1.1, require_cross_domain=False, max_chunks=2,
        )
        return confidence

    @staticmethod
    def _calibrate_hybrid_confidence(scores: Sequence[float]) -> List[float]:
        """Rescale Hybrid's fused ranking scores onto roughly the same
        footing as Standard's calibrated TF-IDF scores, so confidence
        numbers (and the router) aren't comparing apples to oranges.

        Hybrid's ranking is a mix of RRF fusion scores (~0.01-0.05 for a
        couple of fused rankings at rrf_k=60) and, for the top
        CROSS_ENCODER_RERANK_TOP_K items, an already-0-1 cross-encoder
        sigmoid score. A single fixed multiplier tuned for RRF's scale
        (e.g. "x20, then clip to 1.0") pushes the already-well-scaled
        cross-encoder scores straight into the clip, which hides the
        mismatch rather than fixing it, and it silently breaks if rrf_k
        or the number of fused rankings ever changes. Reusing the same
        score/(score+pivot) squashing _calibrate_lexical_scores already
        applies to Standard -- just with a pivot sized for RRF's range --
        keeps ordering, self-normalises regardless of how many rankings
        were fused, and never over/undershoots 0-1. It's a heuristic, not
        a rigorous calibration; treat ROUTER_* thresholds as tunable.
        """
        if not scores:
            return []
        pivot = 0.03  # ballpark for a single fused RRF list at rrf_k=60
        return [score / (score + pivot) if score > 0 else 0.0 for score in scores]

    def _should_use_hybrid(self, query: str) -> Dict[str, Any]:
        """Decide whether Hybrid is likely to outperform Standard for
        `query`, returning the full decision trail (not just a bool) so
        compare() can show its work via router_decision.

        Two hard early-exits cover the unambiguous cases cheaply:
          - Standard confidence already high (> 0.75)  -> Standard.
          - Standard confidence clearly weak  (< 0.40) -> Hybrid.
        Only the ambiguous 0.40-0.75 band falls through to a weighted
        blend of cross-domain + query-complexity signals.

        These early-exits are checked BEFORE the weighted formula rather
        than folded into it, because folding them in creates a real
        contradiction: a confidence of 0.30 should trigger Hybrid per the
        "< 0.40" rule, but the weighted formula alone
        (0.40*(1-0.30) = 0.28, plus 0 for the other two terms) evaluates
        to 0.28 < 0.50 and would recommend Standard -- the opposite of
        the intended rule. Checking the hard bounds first removes that
        contradiction, and also skips the extra work (partition lookup,
        entity extraction) in the clear-cut cases.
        """
        standard_confidence = self._cheap_standard_confidence(query)

        if standard_confidence > self.ROUTER_STANDARD_CONFIDENCE_HIGH:
            return {
                "chosen": "standard",
                "standard_confidence": standard_confidence,
                "cross_domain": False,
                "complexity_score": 0.0,
                "hybrid_score": 0.0,
                "reason": (
                    f"Standard confidence was high ({standard_confidence:.2f} > "
                    f"{self.ROUTER_STANDARD_CONFIDENCE_HIGH}); Hybrid not needed."
                ),
            }
        if standard_confidence < self.ROUTER_STANDARD_CONFIDENCE_LOW:
            return {
                "chosen": "hybrid",
                "standard_confidence": standard_confidence,
                "cross_domain": None,
                "complexity_score": None,
                "hybrid_score": 1.0,
                "reason": (
                    f"Standard confidence was weak ({standard_confidence:.2f} < "
                    f"{self.ROUTER_STANDARD_CONFIDENCE_LOW}); escalating to Hybrid."
                ),
            }

        active_partitions = self._get_active_partitions(query)
        cross_domain = len(active_partitions) >= 2
        term_count = len(query_terms(query))
        entity_count = len(extract_entities(query))
        lower = query.lower()
        comparison_words = sum(
            word in lower for word in
            ("compare", "contrast", "versus", "vs", "between", "difference", "similar")
        )
        complexity_score = min(1.0, (term_count + entity_count + comparison_words) / 15)

        hybrid_score = (
            0.40 * (1 - standard_confidence)
            + 0.30 * (1.0 if cross_domain else 0.0)
            + 0.30 * complexity_score
        )
        chosen = "hybrid" if hybrid_score > self.ROUTER_HYBRID_SCORE_THRESHOLD else "standard"
        return {
            "chosen": chosen,
            "standard_confidence": standard_confidence,
            "cross_domain": cross_domain,
            "complexity_score": complexity_score,
            "hybrid_score": hybrid_score,
            "reason": (
                f"Mixed signal (confidence {standard_confidence:.2f}, cross-domain "
                f"{cross_domain}, complexity {complexity_score:.2f}) -> hybrid_score "
                f"{hybrid_score:.2f} {'>' if chosen == 'hybrid' else '<='} "
                f"{self.ROUTER_HYBRID_SCORE_THRESHOLD}"
            ),
        }

    # -------------------------------------------------------------------------
    # Ranking and fusion
    # -------------------------------------------------------------------------
    @staticmethod
    def _calibrate_lexical_scores(ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        pivot = 0.10
        return [(chunk_id, score / (score + pivot) if score > 0 else 0.0) for chunk_id, score in ranked]

    def _fuse_rankings(self, *rankings: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not rankings:
            return []
        non_empty = [r for r in rankings if r]
        if not non_empty:
            return []
        if len(non_empty) == 1:
            return non_empty[0]
        rrf_scores: Dict[str, float] = defaultdict(float)
        rrf_k = 60
        for ranking in non_empty:
            for position, (chunk_id, _) in enumerate(ranking, start=1):
                rrf_scores[chunk_id] += 1.0 / (rrf_k + position)
        return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    def _rank_candidates(
        self, query: str, candidate_ids: Optional[Sequence[str]], use_hybrid: bool,
        partition_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        if self.lazy:
            return self._rank_candidates_disk(query, use_hybrid, partition_id)
        return self._rank_candidates_in_memory(query, candidate_ids, use_hybrid)

    def _rank_candidates_in_memory(
        self, query: str, candidate_ids: Sequence[str], use_hybrid: bool
    ) -> List[Tuple[str, float]]:
        """Original small-corpus ranking path: everything already resident
        in self.document_tokens/self.entity_graph/embedding index. Used
        whenever processor.lazy is False -- completely unchanged behavior.
        """
        terms = query_terms(query)
        documents = {
            chunk_id: self.document_tokens[chunk_id]
            for chunk_id in candidate_ids
            if chunk_id in self.document_tokens
        }
        lexical = self._calibrate_lexical_scores(vector_rank(documents, self.idf, terms))
        if not use_hybrid:
            return lexical
        rankings = [lexical]
        graph = self.entity_graph.rank(query, documents.keys())
        if graph:
            rankings.append(graph)
        if self.embedding_ranker.model is not None and len(candidate_ids) > 0:
            emb_rank = self.embedding_ranker.rank(query, candidate_ids)
            if emb_rank:
                max_score = max(s for _, s in emb_rank) or 1.0
                emb_rank = [(cid, s / max_score) for cid, s in emb_rank]
                rankings.append(emb_rank)
        fused = self._fuse_rankings(*rankings)
        if self.cross_encoder.model is not None and len(fused) > 3:
            top_k = fused[:CROSS_ENCODER_RERANK_TOP_K]
            top_k_texts = [(cid, self.processor.chunk_texts.get(cid, "")) for cid, _ in top_k]
            reranked_raw = self.cross_encoder.rerank(query, top_k_texts)
            if reranked_raw:
                reranked = [(cid, 1.0 / (1.0 + math.exp(-score))) for cid, score in reranked_raw]
                reranked_ids = {cid for cid, _ in reranked}
                remainder = [item for item in fused if item[0] not in reranked_ids]
                fused = sorted(reranked + remainder, key=lambda item: item[1], reverse=True)
        return fused

    LEXICAL_CANDIDATE_LIMIT = 30  # rows pulled from FTS5/entity_search per query, not the whole corpus

    def _rank_candidates_disk(
        self, query: str, use_hybrid: bool, partition_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Large-corpus ranking path: lexical via FTS5, graph via the
        chunk_entities SQL table -- both disk-backed, so this touches at
        most LEXICAL_CANDIDATE_LIMIT rows regardless of corpus size.
        Cross-encoder/embedding reranking are deliberately NOT run here;
        they're the tier-4 last resort (see _last_resort_rerank), applied
        only to this method's own output when its confidence is low,
        never to the full corpus.
        """
        terms = query_terms(query)
        lexical_raw = self.db.fts_search(terms, partition_id=partition_id, limit=self.LEXICAL_CANDIDATE_LIMIT)
        lexical = self._calibrate_lexical_scores(lexical_raw)
        if not use_hybrid:
            return lexical
        rankings = [lexical]
        entities = extract_entities(query)
        if entities:
            graph = self.db.entity_search(entities, partition_id=partition_id, limit=self.LEXICAL_CANDIDATE_LIMIT)
            if graph:
                rankings.append(graph)
        return self._fuse_rankings(*rankings)

    def _last_resort_rerank(self, query: str, ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Tier 4: cross-encoder + on-demand embedding reranking, applied
        only to Hybrid's own top candidates (LAST_RESORT_CANDIDATE_POOL of
        them, never the full corpus) and only when Hybrid's own
        FTS5+graph confidence wasn't good enough on its own. Embeddings
        are computed fresh for just this small set -- no persistent
        full-corpus embedding index is built or held in RAM.
        """
        if not ranked:
            return ranked
        top = ranked[: self.LAST_RESORT_CANDIDATE_POOL]
        chunk_ids = [cid for cid, _ in top]
        texts = self.db.get_chunk_texts(chunk_ids)
        if not texts:
            return ranked
        rankings = [top]
        if self.embedding_ranker.model is not None:
            emb_rank = self.embedding_ranker.rank_on_demand(query, texts)
            if emb_rank:
                max_score = max(s for _, s in emb_rank) or 1.0
                rankings.append([(cid, s / max_score) for cid, s in emb_rank])
        fused = self._fuse_rankings(*rankings)
        if self.cross_encoder.model is not None and len(fused) > 3:
            top_k = fused[:CROSS_ENCODER_RERANK_TOP_K]
            top_k_texts = [(cid, texts.get(cid, "")) for cid, _ in top_k]
            reranked_raw = self.cross_encoder.rerank(query, top_k_texts)
            if reranked_raw:
                reranked = [(cid, 1.0 / (1.0 + math.exp(-score))) for cid, score in reranked_raw]
                reranked_ids = {cid for cid, _ in reranked}
                remainder = [item for item in fused if item[0] not in reranked_ids]
                fused = sorted(reranked + remainder, key=lambda item: item[1], reverse=True)
        # Anything outside the reranked top pool keeps its original
        # (lower-confidence) position, appended after the reranked items.
        reranked_ids = {cid for cid, _ in fused}
        remainder = [item for item in ranked if item[0] not in reranked_ids]
        return fused + remainder

    WEAK_TOP_SCORE = 0.001

    def _get_cross_domain_rankings(
        self, query: str, use_hybrid: bool = True
    ) -> List[Tuple[str, float]]:
        if self.lazy:
            return self._get_cross_domain_rankings_disk(query, use_hybrid)
        return self._get_cross_domain_rankings_in_memory(query, use_hybrid)

    def _get_cross_domain_rankings_disk(
        self, query: str, use_hybrid: bool = True
    ) -> List[Tuple[str, float]]:
        """Disk-backed equivalent of the in-memory cross-domain blend:
        rank within each active partition via a partition-filtered SQL
        query (own_pool), fall back to an unfiltered (global) query only
        if a partition's own results are weak -- same logic as the
        in-memory version, just backed by fts_search/entity_search
        instead of materialised chunk-id pools.
        """
        active_partitions = self._get_active_partitions(query)
        if not active_partitions:
            return self._rank_candidates(query, None, use_hybrid)
        by_partition: Dict[str, List[Tuple[str, float]]] = {}
        for partition_id in active_partitions:
            ranking = self._rank_candidates(query, None, use_hybrid, partition_id=partition_id)
            if not ranking or ranking[0][1] <= self.WEAK_TOP_SCORE:
                fallback = self._rank_candidates(query, None, use_hybrid)  # unfiltered = global
                ranking = self._merge_rankings_unique(ranking, fallback)
            if ranking:
                by_partition[partition_id] = ranking
        if not by_partition:
            return self._rank_candidates(query, None, use_hybrid)

        seeds: List[Tuple[str, float]] = []
        selected: Set[str] = set()
        for partition_id, ranking in by_partition.items():
            for chunk_id, score in ranking[:MIN_CHUNKS_PER_ACTIVE_PARTITION]:
                if chunk_id not in selected:
                    seeds.append((chunk_id, score))
                    selected.add(chunk_id)
        remaining = [
            item
            for ranking in by_partition.values()
            for item in ranking
            if item[0] not in selected
        ]
        remaining.sort(key=lambda item: item[1], reverse=True)
        return seeds + remaining

    def _get_cross_domain_rankings_in_memory(
        self, query: str, use_hybrid: bool = True
    ) -> List[Tuple[str, float]]:
        active_partitions = self._get_active_partitions(query)
        if not active_partitions:
            return self._rank_candidates(query, list(self.processor.chunk_texts), use_hybrid)
        by_partition: Dict[str, List[Tuple[str, float]]] = {}
        for partition_id in active_partitions:
            own_pool = self.partition_chunks.get(partition_id, [])
            ranking: List[Tuple[str, float]] = (
                self._rank_candidates(query, own_pool, use_hybrid) if own_pool else []
            )
            if not ranking or ranking[0][1] <= self.WEAK_TOP_SCORE:
                global_pool = [cid for cid in self.processor.chunk_texts if cid not in own_pool]
                fallback = self._rank_candidates(query, global_pool, use_hybrid)
                ranking = self._merge_rankings_unique(ranking, fallback)
            if ranking:
                by_partition[partition_id] = ranking
        if not by_partition:
            return self._rank_candidates(query, list(self.processor.chunk_texts), use_hybrid)

        seeds: List[Tuple[str, float]] = []
        selected: Set[str] = set()
        for partition_id, ranking in by_partition.items():
            for chunk_id, score in ranking[:MIN_CHUNKS_PER_ACTIVE_PARTITION]:
                if chunk_id not in selected:
                    seeds.append((chunk_id, score))
                    selected.add(chunk_id)
        remaining = [
            item
            for ranking in by_partition.values()
            for item in ranking
            if item[0] not in selected
        ]
        remaining.sort(key=lambda item: item[1], reverse=True)
        return seeds + remaining

    @staticmethod
    def _merge_rankings_unique(
        primary: List[Tuple[str, float]], fallback: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        best: Dict[str, float] = {}
        for chunk_id, score in primary + fallback:
            if chunk_id not in best or score > best[chunk_id]:
                best[chunk_id] = score
        return sorted(best.items(), key=lambda item: item[1], reverse=True)

    # -------------------------------------------------------------------------
    # Confidence, context, validation
    # -------------------------------------------------------------------------
    def _get_texts_batch(self, chunk_ids: Sequence[str]) -> Dict[str, str]:
        """Fetch text for a small, specific set of chunk_ids (never the
        whole corpus) -- from RAM in the small-corpus case, from the
        database in lazy/large-corpus mode."""
        if self.lazy:
            return self.db.get_chunk_texts(chunk_ids)
        return {cid: self.processor.chunk_texts[cid] for cid in chunk_ids if cid in self.processor.chunk_texts}

    def _get_metadata_batch(self, chunk_ids: Sequence[str]) -> Dict[str, ChunkMetadata]:
        """Same idea as _get_texts_batch, for ChunkMetadata."""
        if self.lazy:
            return self.db.get_chunk_metadata_batch(chunk_ids)
        return {cid: self.processor.chunks_metadata[cid] for cid in chunk_ids if cid in self.processor.chunks_metadata}

    def _calculate_confidence(
        self,
        chunk_ids: Sequence[str],
        normalised_scores: Sequence[float],
        require_cross_domain: bool,
    ) -> float:
        if not chunk_ids or not normalised_scores:
            return 0.0
        strengths = [max(0.0, min(1.0, score)) for score in normalised_scores]
        average_strength = sum(strengths) / len(strengths)
        weakest_evidence = min(strengths)
        metadata_batch = self._get_metadata_batch(chunk_ids)
        partitions = {meta.partition_id for meta in metadata_batch.values()}
        if require_cross_domain and len(partitions) < 2:
            return 0.0
        cross_domain_bonus = 0.10 if require_cross_domain and len(partitions) >= 2 else 0.0
        return max(
            0.0,
            min(1.0, (0.65 * average_strength) + (0.25 * weakest_evidence) + cross_domain_bonus),
        )

    @staticmethod
    def _truncate_on_sentence_boundary(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        window = text[:limit]
        best_end = -1
        for match in re.finditer(r"[.!?](?:\s|$)", window):
            best_end = match.end()
        if best_end >= int(limit * 0.4):
            return window[:best_end].strip()
        return window.rsplit(" ", 1)[0]

    def _build_context(self, chunk_ids: Sequence[str]) -> str:
        texts = self._get_texts_batch(chunk_ids)
        metadata_batch = self._get_metadata_batch(chunk_ids)
        sections = []
        for chunk_id in chunk_ids:
            text = texts.get(chunk_id, "")
            metadata = metadata_batch.get(chunk_id)
            if not text or not metadata:
                continue
            partition = metadata.partition_id.replace("partition_", "")
            excerpt = self._truncate_on_sentence_boundary(text, CONTEXT_CHARS_PER_CHUNK)
            sections.append(f"[{chunk_id}|{partition}]\n{excerpt}")
        return "\n\n".join(sections)

    def _source_summary(self, chunk_ids: Sequence[str]) -> List[str]:
        metadata_batch = self._get_metadata_batch(chunk_ids)
        sources = []
        for chunk_id in chunk_ids:
            metadata = metadata_batch.get(chunk_id)
            if not metadata:
                continue
            sources.append(
                f"{chunk_id} | {Path(metadata.file_path).name} | "
                f"{metadata.partition_id.replace('partition_', '')}"
            )
        return sources

    def _validate_answer_quality(
        self,
        query: str,
        context: str,
        chunk_ids: Sequence[str],
        require_cross_domain: bool,
    ) -> bool:
        if not context.strip() or not chunk_ids:
            return False
        if require_cross_domain:
            metadata_batch = self._get_metadata_batch(chunk_ids)
            partitions = {meta.partition_id for meta in metadata_batch.values()}
            if len(partitions) < 2:
                return False
            query_entities = extract_entities(query)
            if query_entities:
                texts_batch = self._get_texts_batch(chunk_ids)
                for partition_id in partitions:
                    partition_text = " ".join(
                        texts_batch.get(chunk_id, "")
                        for chunk_id in chunk_ids
                        if metadata_batch.get(chunk_id)
                        and metadata_batch[chunk_id].partition_id == partition_id
                    ).lower()
                    if not any(entity.lower() in partition_text for entity in query_entities):
                        return False
        terms = set(query_terms(query))
        if not terms:
            return True
        context_tokens = {stem_token(token) for token in re.findall(r"\b[\w-]+\b", context)}
        coverage = sum(term in context_tokens for term in terms) / len(terms)
        coverage_threshold = 0.45 if require_cross_domain else 0.55
        if coverage < coverage_threshold:
            return False
        return True

    # -------------------------------------------------------------------------
    # Intelligent retrieval and answer generation
    # -------------------------------------------------------------------------
    def _intelligent_retrieve(
        self,
        ranked_chunks: Sequence[Tuple[str, float]],
        query: str,
        confidence_threshold: float,
        require_cross_domain: bool,
        max_chunks: int = MAX_CHUNKS,
    ) -> Tuple[List[str], float]:
        ranked_chunks = list(ranked_chunks[:max_chunks])
        if not ranked_chunks:
            return [], 0.0
        selected_ids: List[str] = []
        selected_scores: List[float] = []
        for chunk_id, score in ranked_chunks:
            if chunk_id in selected_ids:
                continue
            selected_ids.append(chunk_id)
            selected_scores.append(max(0.0, min(1.0, score)))
            if len(selected_ids) < min(2, len(ranked_chunks)):
                continue
            confidence = self._calculate_confidence(
                selected_ids, selected_scores, require_cross_domain
            )
            context = self._build_context(selected_ids)
            quality_ok = self._validate_answer_quality(
                query, context, selected_ids, require_cross_domain
            )
            if confidence >= confidence_threshold and quality_ok:
                return selected_ids, confidence
        return selected_ids, self._calculate_confidence(
            selected_ids, selected_scores, require_cross_domain
        )

    def _calculate_chunk_relevance(self, query: str, chunk_ids: Sequence[str]) -> float:
        terms = query_terms(query)
        if not terms or not chunk_ids:
            return 0.0
        texts_batch = self._get_texts_batch(chunk_ids)
        per_chunk = []
        for chunk_id in chunk_ids:
            text = texts_batch.get(chunk_id, "").lower()
            if not text:
                continue
            text_tokens = {stem_token(token) for token in re.findall(r"\b[\w-]+\b", text)}
            covered = sum(term in text_tokens for term in set(terms)) / len(set(terms))
            entity_bonus = 0.1 * len(extract_entities(query) & extract_entities(text))
            per_chunk.append(min(1.0, covered + entity_bonus))
        return 100.0 * (sum(per_chunk) / len(per_chunk)) if per_chunk else 0.0

    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        terms = set(query_terms(query))
        if not terms or not answer:
            return 0.0
        answer_tokens = {stem_token(token) for token in re.findall(r"\b[\w-]+\b", answer.lower())}
        coverage = sum(term in answer_tokens for term in terms) / len(terms)
        return 100.0 * min(1.0, coverage)

    def _local_answer(self, query: str, context: str, method: str) -> str:
        if not context:
            return "No retrieved context is available."
        terms = query_terms(query)
        sentences = re.split(r"(?<=[.!?])\s+", context)
        relevant = [
            sentence.strip() for sentence in sentences
            if len(sentence.strip()) > 40 and any(term in sentence.lower() for term in terms)
        ]
        if not relevant:
            return "Relevant evidence was retrieved, but no concise local extract was found."
        return f"{method} local evidence summary:\n" + "\n".join(f"- {sentence}" for sentence in relevant[:4])

    def _generate_answer(self, query: str, context: str, method: str, provider_configs: List[ProviderConfig]) -> Tuple[str, Dict[str, Any]]:
        if not provider_configs:
            return self._local_answer(query, context, method), {"provider": None, "notice": ""}
        if context:
            prompt = (
                "You are answering a technical question using retrieved evidence "
                "from research papers.\n\n"
                f"Evidence:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Write a clear, complete answer. Ground it in the evidence above "
                "and cite the relevant [chunk_id] next to any claim it supports. "
                "If the evidence only partially covers the question, you may use "
                "your own general knowledge to fill gaps or add context -- but "
                "make it clear when you're doing so (e.g. 'more generally, ...') "
                "rather than presenting it as coming from the cited papers."
            )
        else:
            prompt = f"Q: {query}\nGive a concise answer and state uncertainty where relevant."
        failures = []
        for config in provider_configs:
            try:
                input_tokens = estimate_tokens(prompt)
                if config.name in self.db.CREDIT_PROVIDERS:
                    allowed, reason = self.db.can_generate(
                        config.name, input_tokens, MAX_ANSWER_TOKENS
                    )
                    if not allowed:
                        failures.append(reason)
                        continue
                if config.name == "claude":
                    if anthropic is None:
                        raise RuntimeError("install anthropic")
                    response = anthropic.Anthropic(api_key=config.api_key, timeout=30.0, max_retries=1).messages.create(
                        model=config.model, max_tokens=MAX_ANSWER_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    answer = "".join(block.text for block in response.content if block.type == "text")
                    actual_input = int(getattr(response.usage, "input_tokens", input_tokens) or input_tokens)
                    actual_output = int(getattr(response.usage, "output_tokens", estimate_tokens(answer)) or estimate_tokens(answer))
                elif config.name == "gemini":
                    if genai is None:
                        raise RuntimeError("install google-genai")
                    response = genai.Client(api_key=config.api_key, http_options={"timeout": 30_000}).models.generate_content(
                        model=config.model, contents=prompt,
                    )
                    answer = response.text or ""
                    usage = getattr(response, "usage_metadata", None)
                    actual_input = int(getattr(usage, "prompt_token_count", input_tokens) or input_tokens)
                    actual_output = int(getattr(usage, "candidates_token_count", estimate_tokens(answer)) or estimate_tokens(answer))
                else:
                    if OpenAI is None:
                        raise RuntimeError("install openai")
                    kwargs: Dict[str, Any] = {"api_key": config.api_key, "timeout": 30.0, "max_retries": 1}
                    if config.base_url:
                        kwargs["base_url"] = config.base_url
                    client = OpenAI(**kwargs)
                    response = client.chat.completions.create(
                        model=config.model, max_completion_tokens=MAX_ANSWER_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    answer = response.choices[0].message.content or ""
                    usage = response.usage
                    actual_input = int(getattr(usage, "prompt_tokens", input_tokens) or input_tokens)
                    actual_output = int(getattr(usage, "completion_tokens", estimate_tokens(answer)) or estimate_tokens(answer))
                if clean_text(answer):
                    cost = 0.0
                    if config.name in self.db.CREDIT_PROVIDERS:
                        cost = self.db.record_usage(config.name, config.model, actual_input, actual_output)
                    generation_info = {
                        "provider": config.name,
                        "input_tokens": actual_input,
                        "output_tokens": actual_output,
                        "cost_usd": cost,
                        "notice": "",
                    }
                    return clean_text(answer), generation_info
                raise RuntimeError("empty response")
            except Exception as exc:
                failures.append(f"{config.name}: {exc}")
        notice = "No provider call was made; local evidence extraction was used. " + "; ".join(failures)
        generation_info = {"provider": None, "notice": notice}
        return self._local_answer(query, context, method) + "\n[" + notice + "]", generation_info

    # -------------------------------------------------------------------------
    # Main RAG execution (FORCED HYBRID – no cheap router)
    # -------------------------------------------------------------------------
    def _run_rag(self, query: str, method: str, use_hybrid: bool, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        start = time.perf_counter()
        is_cross_domain = self._is_cross_domain_question(query) if use_hybrid else False
        threshold = self._get_adaptive_threshold(query, is_cross_domain)
        max_chunks = self._max_chunks_for_query(query, is_cross_domain)

        if use_hybrid:
            ranked = self._get_cross_domain_rankings(query, use_hybrid=True)
            # Hybrid's fused (RRF + cross-encoder) scores live on a different
            # scale than Standard's calibrated TF-IDF scores -- calibrate
            # before confidence/threshold comparisons so the two pipelines
            # are judged on comparable footing (see _calibrate_hybrid_confidence).
            calibrated_scores = self._calibrate_hybrid_confidence([score for _, score in ranked])
            ranked = list(zip((chunk_id for chunk_id, _ in ranked), calibrated_scores))
            chunk_ids, confidence = self._intelligent_retrieve(
                ranked, query, threshold, is_cross_domain, max_chunks
            )
            retrieval_path = "hybrid"
            # Tier 4 (large-corpus mode only): the disk-backed FTS5+graph
            # Hybrid tier above never touches cross-encoder/embeddings --
            # that's what keeps its latency low across the whole query set.
            # If ITS OWN confidence still isn't good enough, escalate just
            # this query's own top candidates (never the full corpus) to
            # cross-encoder + on-demand embedding reranking, as a rare
            # last resort rather than a per-query default.
            if self.lazy and confidence < self.ROUTER_LAST_RESORT_THRESHOLD and (
                self.cross_encoder.model is not None or self.embedding_ranker.model is not None
            ):
                reranked = self._last_resort_rerank(query, ranked)
                escalated_ids, escalated_confidence = self._intelligent_retrieve(
                    reranked, query, threshold, is_cross_domain, max_chunks
                )
                if escalated_confidence > confidence:
                    chunk_ids, confidence = escalated_ids, escalated_confidence
                    retrieval_path = "hybrid_last_resort"
        else:
            # Standard RAG (lexical only: TF-IDF in-memory, or FTS5 disk-backed)
            candidate_ids = None if self.lazy else list(self.processor.chunk_texts)
            ranked = self._rank_candidates(query, candidate_ids, False)
            chunk_ids, confidence = self._intelligent_retrieve(
                ranked, query, threshold, is_cross_domain, max_chunks
            )
            retrieval_path = "standard_rag"

        context = self._build_context(chunk_ids)
        answer, generation_info = self._generate_answer(query, context, method, provider_configs)
        latency = time.perf_counter() - start
        quality_validated = self._validate_answer_quality(
            query, context, chunk_ids, is_cross_domain
        )

        return {
            "answer": answer,
            "chunks_retrieved": len(chunk_ids),
            "tokens_used": estimate_tokens(context),
            "latency": latency,
            "chunk_ids": chunk_ids,
            "relevance_score": self._calculate_chunk_relevance(query, chunk_ids),
            "answer_relevance": self._calculate_answer_relevance(query, answer),
            "confidence_score": confidence,
            "threshold_used": threshold,
            "quality_validated": quality_validated,
            "cross_domain": is_cross_domain,
            "sources": self._source_summary(chunk_ids),
            "generation": generation_info,
        }

    def _hybrid_rag(self, query: str, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        return self._run_rag(query, "Hybrid RAG", use_hybrid=True, provider_configs=provider_configs)

    def _standard_rag(self, query: str, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        return self._run_rag(query, "Standard RAG", use_hybrid=False, provider_configs=provider_configs)

    def _direct_ai(self, query: str, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        start = time.perf_counter()
        answer, generation_info = self._generate_answer(query, "", "Direct AI", provider_configs)
        return {
            "answer": answer,
            "chunks_retrieved": 0,
            "tokens_used": 0,
            "latency": time.perf_counter() - start,
            "chunk_ids": [],
            "relevance_score": 0.0,
            "answer_relevance": self._calculate_answer_relevance(query, answer),
            "confidence_score": 0.0,
            "threshold_used": 0.0,
            "quality_validated": False,
            "cross_domain": False,
            "sources": [],
            "generation": generation_info,
        }

    def compare(self, query: str, provider: Optional[str] = None, api_key: Optional[str] = None,
                include_direct_ai: bool = True, force_hybrid: bool = False) -> Dict[str, Dict[str, Any]]:
        """Runs Standard and Hybrid RAG (both, unconditionally -- Hybrid is
        NOT routed/skipped here; see the hardcoded 'Forced Hybrid' decision
        below). This is deliberate for evaluation: every question gets a
        real answer from both pipelines so they can be compared fairly,
        rather than the smart-router behavior (used elsewhere) where
        Hybrid gets skipped whenever Standard looks confident enough.

        include_direct_ai=False skips the third, no-retrieval "ask the
        model directly" generation call -- a real, separate API call on
        every question, wasted if the caller (e.g. a batch eval script)
        never reads results["direct_ai"].
        """
        provider_configs = (
            available_provider_configs(provider, key_override=api_key)
            if provider is not None
            else self.provider_configs
        )
        if force_hybrid:
            decision = {"chosen": "hybrid", "standard_confidence": 0.0, "reason": "Forced Hybrid (eval mode)"}
        else:
            decision = self._should_use_hybrid(query)
        standard = self._standard_rag(query, provider_configs)

        if decision["chosen"] == "hybrid":
            hybrid = self._hybrid_rag(query, provider_configs)
            decision["hybrid_confidence_calibrated"] = hybrid["confidence_score"]
            decision["standard_was_used"] = False
            decision["hybrid_was_skipped"] = False
        else:
            hybrid = dict(standard)
            hybrid["generation"] = dict(standard.get("generation", {}))
            decision["hybrid_confidence_calibrated"] = None
            decision["standard_was_used"] = True
            decision["hybrid_was_skipped"] = True

        direct = self._direct_ai(query, provider_configs) if include_direct_ai else None
        saved = max(0, standard["tokens_used"] - hybrid["tokens_used"])
        hybrid["tokens_saved_vs_standard"] = saved
        hybrid["token_savings_percent_vs_standard"] = (
            100.0 * saved / standard["tokens_used"]
            if standard["tokens_used"] else 0.0
        )
        hybrid["router_decision"] = decision
        results = {
            "hybrid_rag": hybrid,
            "standard_rag": standard,
        }
        if direct is not None:
            # Omitted entirely (not set to None) when skipped: both
            # save_comparison and format_comparison iterate results.items()
            # assuming every value is a populated result dict -- a None
            # value would crash both on the first .get()/[...] access.
            results["direct_ai"] = direct
        self.db.save_comparison(query, results)
        print(f"[router] {decision['reason']}")
        return results

    def compare_single(self, query: str, method: str = "hybrid_rag",
                       provider: str = "auto", api_key: Optional[str] = None) -> Dict[str, Any]:
        provider_configs = available_provider_configs(provider, key_override=api_key)
        if method == "hybrid_rag":
            return self._hybrid_rag(query, provider_configs)
        elif method == "standard_rag":
            return self._standard_rag(query, provider_configs)
        elif method == "direct_ai":
            return self._direct_ai(query, provider_configs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def test_provider_connection(self, provider: str, api_key: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        provider_configs = available_provider_configs(provider, key_override=api_key)
        if not provider_configs:
            return "", {"provider": None, "notice": f"No usable configuration for '{provider}' (no API key found, typed or in env)."}
        return self._generate_answer("Reply with exactly: OK", "", "connection test", provider_configs)

    def format_comparison(self, results: Dict[str, Dict[str, Any]]) -> str:
        lines = [
            "\n" + "=" * 88,
            "COMPARISON RESULTS (SMART ROUTER v23)",
            "=" * 88,
        ]
        router = results.get("hybrid_rag", {}).get("router_decision")
        if router:
            lines.append(
                f"Router: {router['chosen'].title()} "
                f"(confidence: {router['standard_confidence']:.2f}, "
                f"cross-domain: {'Yes' if router.get('cross_domain') else 'No'}, "
                f"complexity: {router.get('complexity_score') or 0:.2f})"
            )
            lines.append(router["reason"])
            if router.get("hybrid_was_skipped"):
                lines.append("Hybrid: NOT RUN (Standard was sufficient)")
            lines.append("-" * 88)
        lines.extend([
            "NOTE: 'Chunk relevance', 'confidence', and 'quality' are grounding",
            "metrics -- they measure whether retrieved evidence overlaps the query and",
            "spans the right domains. They are legitimately 0/no for Direct AI, since it",
            "retrieves nothing. Use 'Answer relevance' (query-term coverage of the",
            "actual answer text) for a metric that's computed the same way across all",
            "pipelines.",
            "-" * 88,
            "Method        Latency    Ctx tokens  Chunks  Chunk rel  Answer rel  Confidence  Quality",
            "-" * 88,
        ])
        for method, result in results.items():
            lines.append(
                f"{method:<13} {result['latency']:>7.2f}s"
                f" {result['tokens_used']:>11}"
                f" {result['chunks_retrieved']:>7}"
                f" {result['relevance_score']:>10.1f}"
                f" {result.get('answer_relevance', 0.0):>11.1f}"
                f" {result['confidence_score']:>11.2f}"
                f"  {'yes' if result['quality_validated'] else 'no'}"
            )
        hybrid = results["hybrid_rag"]
        lines.append("-" * 88)
        lines.append(
            "Hybrid context-token savings vs Standard: "
            f"{hybrid.get('tokens_saved_vs_standard', 0)} "
            f"({hybrid.get('token_savings_percent_vs_standard', 0.0):.1f}%)"
        )
        for method, result in results.items():
            lines.extend([
                "\n" + method.upper(),
                f"Threshold: {result['threshold_used']:.2f} | "
                f"Chunk IDs: {', '.join(result['chunk_ids']) or 'none'}",
                "Sources: " + ("; ".join(result.get("sources", [])) or "none"),
                result["answer"],
            ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export and CLI
# ---------------------------------------------------------------------------
def export_markdown(query: str, results: Dict[str, Dict[str, Any]]) -> Path:
    output_dir = Path("rag_exports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.md"
    lines = [
        f"# Forced Hybrid RAG comparison (v23)\n\nQuestion: {query}\n",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
    ]
    for method, result in results.items():
        lines.extend([
            f"## {method}\n",
            f"- Latency: {result['latency']:.2f}s",
            f"- Estimated context tokens: {result['tokens_used']}",
            f"- Chunks: {result['chunks_retrieved']}",
            f"- Chunk relevance: {result['relevance_score']:.1f}",
            f"- Answer relevance: {result.get('answer_relevance', 0.0):.1f}",
            f"- Confidence: {result['confidence_score']:.2f}",
            f"- Threshold: {result['threshold_used']:.2f}",
            f"- Quality validated: {result['quality_validated']}",
            f"- Chunk IDs: {', '.join(result['chunk_ids']) or 'none'}\n",
            f"### Answer\n\n{result['answer']}\n",
        ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_large_corpus_engine(db_path: str = DATABASE_PATH, database_url: str = None, provider: str = "auto", api_key: Optional[str] = None) -> Tuple[DatabaseManager, DocumentProcessor, "ComparisonEngine"]:
    """Construct db/processor/engine in large-corpus (memory-bounded) mode:
    DocumentProcessor never loads the full corpus into RAM, and
    ComparisonEngine retrieves via disk-backed FTS5/entity-table queries
    instead of in-memory TF-IDF/entity-graph structures. Use this for
    corpora too large to hold as Python objects (e.g. ~1.9M chunks);
    use plain DocumentProcessor(db) + ComparisonEngine(...) (main()'s
    default) for small corpora where the original in-RAM path is fine
    and slightly faster.
    """
    db = DatabaseManager(db_path=db_path, database_url=database_url)
    processor = DocumentProcessor(db, lazy=True)
    engine = ComparisonEngine(processor, db, provider=provider, api_key=api_key)
    return db, processor, engine


def print_status(db: DatabaseManager, processor: DocumentProcessor) -> None:
    documents = db.get_all_documents()
    if processor.lazy:
        total_chunks = sum(db.get_partition_counts().values())
    else:
        total_chunks = len(processor.chunk_texts)
    print(f"\nLoaded documents: {len(documents)} | total chunks: {total_chunks}")
    for document in documents:
        print(
            f"  - {document['file_name']} | {document['partition_id']} | "
            f"{document['chunk_count']} chunks"
        )


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    print("=" * 72)
    print("FORCED HYBRID RAG v23 (AUTO‑PARTITIONING + DUMP‑AND‑GO)")
    print("TF-IDF + Entity Graph/RRF + Embeddings + Cross-Encoder (if available)")
    print("=" * 72)
    db = DatabaseManager()
    processor = DocumentProcessor(db)
    print_status(db, processor)
    provider = choose_provider()
    while True:
        print("\n1) Add papers with file picker")
        print("2) Add papers by path")
        print("3) Ask and compare")
        print("4) Show status")
        print("5) Quit")
        choice = input("Select (1-5): ").strip()
        if choice == "1":
            files = select_files_dialog()
            if files:
                processor.process_documents(files)
        elif choice == "2":
            raw_paths = input("Comma-separated PDF/TXT/MD/PY/JS/IPYNB/ZIP paths: ").strip()
            if raw_paths:
                processor.process_documents([path.strip() for path in raw_paths.split(",")])
        elif choice == "3":
            if not processor.chunk_texts:
                print("Load at least one document first.")
                continue
            query = input("\nQuestion: ").strip()
            if not query:
                continue
            engine = ComparisonEngine(processor, db, provider)
            results = engine.compare(query)
            print(engine.format_comparison(results))
            if input("Export Markdown report? (y/n): ").strip().lower() == "y":
                path = export_markdown(query, results)
                print(f"Exported: {path.resolve()}")
        elif choice == "4":
            print_status(db, processor)
        elif choice in {"5", "q", "quit", "exit"}:
            print("Goodbye.")
            return
        else:
            print("Please enter a number from 1 to 5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
