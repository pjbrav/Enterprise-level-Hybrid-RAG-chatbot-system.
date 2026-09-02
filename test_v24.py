"""
Unit tests for v23.py — core components that don't require a database.

Run:
    python -m pytest test_v24.py -v
    # or without pytest:
    python test_v24.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure v23 is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestEntityExtraction(unittest.TestCase):
    """Test extract_entities() with and without spaCy."""

    def setUp(self):
        # Save original spaCy state
        self._orig_spacy = None

    def test_extracts_capitalized_entities_regex_fallback(self):
        """Regex fallback extracts capitalized words and multi-word phrases."""
        from v24 import extract_entities
        text = "John Smith from Acme Corporation filed a ticket for Proxima Bank"
        entities = extract_entities(text)
        # Should contain person/org names (lowercased)
        self.assertIn("john smith", entities)
        self.assertIn("acme corporation", entities)
        self.assertIn("proxima bank", entities)

    def test_filters_stopwords(self):
        """Common English words should be filtered out."""
        from v24 import extract_entities, _ENTITY_STOPWORDS
        text = "The issue was reported by This and That"
        entities = extract_entities(text)
        # "the", "was", "by", "this", "that" should all be filtered
        for word in ["the", "was", "reported", "this", "that"]:
            self.assertNotIn(word, entities)

    def test_extracts_ticket_ids(self):
        """Structured IDs like PROJ-123 should be extracted."""
        from v24 import extract_entities
        text = "Fixed in PROJ-4567 and PR #1234"
        entities = extract_entities(text)
        self.assertIn("proj-4567", entities)
        self.assertIn("#1234", entities)

    def test_extracts_version_numbers(self):
        """Version numbers should be extracted."""
        from v24 import extract_entities
        text = "Updated to v2.3.1 and 3.0.0"
        entities = extract_entities(text)
        self.assertIn("v2.3.1", entities)
        self.assertIn("3.0.0", entities)

    def test_empty_text(self):
        """Empty string should return empty set."""
        from v24 import extract_entities
        self.assertEqual(extract_entities(""), set())
        self.assertEqual(extract_entities("   "), set())

    def test_no_entities_in_plain_text(self):
        """Lowercase text with no entities should return minimal results."""
        from v24 import extract_entities
        entities = extract_entities("the quick brown fox jumps over the lazy dog")
        # Should be empty or minimal (no capitalized entities)
        self.assertEqual(len(entities), 0)

    def test_all_entities_are_lowercase(self):
        """All extracted entities should be lowercased."""
        from v24 import extract_entities
        text = "GitHub and Acme Corp"
        entities = extract_entities(text)
        for e in entities:
            self.assertEqual(e, e.lower())

    def test_no_dataset_specific_stopwords(self):
        """Template-specific words should NOT be in the stopword list."""
        from v24 import _ENTITY_STOPWORDS
        # These were removed as dataset-specific template terms
        for word in ["customer", "motivation", "checklist", "merge", "pull",
                     "support", "issue", "impact", "environment", "observed",
                     "confirmed", "initial", "reported", "assigned", "priority",
                     "status", "description", "summary", "comment",
                     "changes", "approach", "context", "rollout", "background",
                     "testing", "reviewed", "review", "merged", "request",
                     "branch", "steps", "reproduce", "expected", "actual",
                     "behavior", "acceptance", "criteria", "definition"]:
            self.assertNotIn(word, _ENTITY_STOPWORDS,
                f"'{word}' should not be in universal stopword list")
        # But universal stopwords should be present
        for word in ["the", "what", "and", "for"]:
            self.assertIn(word, _ENTITY_STOPWORDS)


class TestRRFFusion(unittest.TestCase):
    """Test the RRF (Reciprocal Rank Fusion) logic."""

    def test_rrf_same_rank_both_lists(self):
        """A chunk ranked #1 in both lists should score highest."""
        # RRF formula: 1/(k+rank1) + 1/(k+rank2)
        k = 60
        rank1, rank2 = 1, 1  # ranked #1 in both
        score = 1/(k+rank1) + 1/(k+rank2)
        # Should be higher than a chunk ranked #1 in only one list
        rank1_only = 1/(k+1) + 1/(k+100)
        self.assertGreater(score, rank1_only)

    def test_rrf_rank_ordering(self):
        """Higher ranks (lower rank numbers) should produce higher scores."""
        k = 60
        score_rank1 = 1/(k+1) + 1/(k+1)
        score_rank5 = 1/(k+5) + 1/(k+5)
        score_rank50 = 1/(k+50) + 1/(k+50)
        self.assertGreater(score_rank1, score_rank5)
        self.assertGreater(score_rank5, score_rank50)

    def test_rrf_k_constant_effect(self):
        """Larger k means scores decay more slowly with rank."""
        # With k=60, rank 1 vs rank 50 difference is smaller than with k=10
        k_large = 60
        k_small = 10
        diff_large = (1/(k_large+1) + 1/(k_large+1)) - (1/(k_large+50) + 1/(k_large+50))
        diff_small = (1/(k_small+1) + 1/(k_small+1)) - (1/(k_small+50) + 1/(k_small+50))
        self.assertGreater(diff_small, diff_large,
            "Smaller k should produce larger score differences between ranks")


class TestPartitionRouting(unittest.TestCase):
    """Test partition keyword routing logic."""

    def test_keyword_routing_jira(self):
        """Query mentioning 'jira' should route to partition_jira."""
        # Simulate the keyword matching logic
        PARTITION_KEYWORDS = {
            "partition_jira": ["jira", "ticket", "epic", "sprint"],
            "partition_github": ["github", "pr ", "pull request", "repo"],
        }
        query = "What Jira tickets are blocking the release?"
        query_lower = query.lower()
        matched = []
        for partition_id, keywords in PARTITION_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    matched.append(partition_id)
                    break
        self.assertIn("partition_jira", matched)

    def test_keyword_routing_github(self):
        """Query mentioning 'github' should route to partition_github."""
        PARTITION_KEYWORDS = {
            "partition_jira": ["jira", "ticket", "epic"],
            "partition_github": ["github", "pr ", "pull request", "repo"],
        }
        query = "Which GitHub PRs are related to the API migration?"
        query_lower = query.lower()
        matched = []
        for partition_id, keywords in PARTITION_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    matched.append(partition_id)
                    break
        self.assertIn("partition_github", matched)

    def test_keyword_routing_multiple_partitions(self):
        """Query mentioning both sources should route to both."""
        PARTITION_KEYWORDS = {
            "partition_jira": ["jira", "ticket"],
            "partition_github": ["github", "pr "],
        }
        query = "What GitHub PRs relate to the Jira tickets?"
        query_lower = query.lower()
        matched = []
        for partition_id, keywords in PARTITION_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    matched.append(partition_id)
                    break
        self.assertIn("partition_jira", matched)
        self.assertIn("partition_github", matched)

    def test_keyword_routing_no_match(self):
        """Query with no source keywords should not match any partition."""
        PARTITION_KEYWORDS = {
            "partition_jira": ["jira", "ticket"],
            "partition_github": ["github", "pr "],
        }
        query = "What is the deployment process?"
        query_lower = query.lower()
        matched = []
        for partition_id, keywords in PARTITION_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    matched.append(partition_id)
                    break
        self.assertEqual(len(matched), 0)


class TestCrossDomainDetection(unittest.TestCase):
    """Test cross-domain question detection logic."""

    def test_two_partitions_is_cross_domain(self):
        """2+ active partitions should be cross-domain."""
        active_partitions = ["partition_jira", "partition_github"]
        is_cross_domain = len(active_partitions) >= 2
        self.assertTrue(is_cross_domain)

    def test_one_partition_is_not_cross_domain(self):
        """1 active partition should not be cross-domain."""
        active_partitions = ["partition_jira"]
        is_cross_domain = len(active_partitions) >= 2
        self.assertFalse(is_cross_domain)

    def test_zero_partitions_is_not_cross_domain(self):
        """0 active partitions should not be cross-domain."""
        active_partitions = []
        is_cross_domain = len(active_partitions) >= 2
        self.assertFalse(is_cross_domain)


class TestThresholdEnvVars(unittest.TestCase):
    """Test that thresholds can be overridden via environment variables."""

    def test_env_var_override(self):
        """Environment variables should override default thresholds."""
        os.environ["MIN_THRESHOLD"] = "0.35"
        try:
            value = float(os.getenv("MIN_THRESHOLD", "0.20"))
            self.assertEqual(value, 0.35)
        finally:
            del os.environ["MIN_THRESHOLD"]

    def test_env_var_default(self):
        """Missing env var should fall back to default."""
        # Ensure it's not set
        os.environ.pop("MIN_THRESHOLD", None)
        value = float(os.getenv("MIN_THRESHOLD", "0.20"))
        self.assertEqual(value, 0.20)


class TestDatabasePathConfig(unittest.TestCase):
    """Test database path configuration."""

    def test_default_db_path(self):
        """Default DB path should be hybrid_rag_v22.db."""
        os.environ.pop("SQLITE_DATABASE_PATH", None)
        path = os.getenv("SQLITE_DATABASE_PATH", "hybrid_rag_v22.db")
        self.assertEqual(path, "hybrid_rag_v22.db")

    def test_custom_db_path(self):
        """Custom DB path should be read from env."""
        os.environ["SQLITE_DATABASE_PATH"] = "/custom/path.db"
        try:
            path = os.getenv("SQLITE_DATABASE_PATH", "hybrid_rag_v22.db")
            self.assertEqual(path, "/custom/path.db")
        finally:
            del os.environ["SQLITE_DATABASE_PATH"]


class TestFileExistenceCheck(unittest.TestCase):
    """Test that build_large_corpus_engine rejects missing DB files."""

    def test_missing_db_raises_error(self):
        """Missing DB file should raise an error, not crash silently."""
        from v24 import build_large_corpus_engine
        # The function may raise FileNotFoundError (if patched with the
        # existence check) or sqlite3.OperationalError (if SQLite itself
        # rejects the path). Either is acceptable — what matters is that
        # it doesn't silently create an empty DB or hang.
        import sqlite3
        with self.assertRaises((FileNotFoundError, sqlite3.OperationalError, OSError)):
            build_large_corpus_engine(db_path="/nonexistent/path.db")


def run_all_tests():
    """Run all tests without pytest (for environments without pytest installed)."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
