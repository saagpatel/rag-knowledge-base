"""Tests for benchmark scoring functions."""

from __future__ import annotations

from rag_kb.cli.benchmark import recall_at_k, reciprocal_rank


class TestRecallAtK:
    def test_all_found(self):
        """All expected files found in top-k."""
        retrieved = ["/docs/a.md", "/docs/b.md", "/docs/c.md"]
        expected = ["a.md", "b.md"]
        assert recall_at_k(retrieved, expected, 5) == 1.0

    def test_none_found(self):
        """No expected files found."""
        retrieved = ["/docs/x.md", "/docs/y.md"]
        expected = ["a.md", "b.md"]
        assert recall_at_k(retrieved, expected, 5) == 0.0

    def test_partial(self):
        """Some expected files found."""
        retrieved = ["/docs/a.md", "/docs/x.md", "/docs/y.md"]
        expected = ["a.md", "b.md"]
        assert recall_at_k(retrieved, expected, 5) == 0.5

    def test_empty_expected(self):
        """Empty expected returns 0."""
        retrieved = ["/docs/a.md"]
        assert recall_at_k(retrieved, [], 5) == 0.0

    def test_k_limits_results(self):
        """Only top-k results are considered."""
        retrieved = ["/docs/x.md", "/docs/y.md", "/docs/a.md"]
        expected = ["a.md"]
        assert recall_at_k(retrieved, expected, 2) == 0.0
        assert recall_at_k(retrieved, expected, 3) == 1.0

    def test_substring_matching(self):
        """Substring matching works for partial paths."""
        retrieved = ["/home/user/projects/docs/guide.md"]
        expected = ["guide.md"]
        assert recall_at_k(retrieved, expected, 5) == 1.0


class TestReciprocalRank:
    def test_first_position(self):
        """First result is relevant — MRR = 1.0."""
        retrieved = ["/docs/a.md", "/docs/b.md"]
        expected = ["a.md"]
        assert reciprocal_rank(retrieved, expected) == 1.0

    def test_third_position(self):
        """Relevant result at position 3 — MRR = 1/3."""
        retrieved = ["/docs/x.md", "/docs/y.md", "/docs/a.md"]
        expected = ["a.md"]
        assert abs(reciprocal_rank(retrieved, expected) - 1 / 3) < 0.001

    def test_no_match(self):
        """No relevant results — MRR = 0."""
        retrieved = ["/docs/x.md", "/docs/y.md"]
        expected = ["a.md"]
        assert reciprocal_rank(retrieved, expected) == 0.0

    def test_empty_expected(self):
        """Empty expected returns 0."""
        retrieved = ["/docs/a.md"]
        assert reciprocal_rank(retrieved, []) == 0.0
