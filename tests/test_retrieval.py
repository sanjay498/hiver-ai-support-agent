"""Tests for FAISS Historical Resolution Retrieval."""

import pytest
from src.retrieval.retriever import HistoricalRetriever, RetrievedEvidence


def test_retriever_loading():
    retriever = HistoricalRetriever()
    assert retriever.index is not None
    assert retriever.index.ntotal > 0
    assert len(retriever.metadata) == retriever.index.ntotal


def test_retriever_query():
    retriever = HistoricalRetriever()
    results = retriever.retrieve("Where is my package? Delayed for days.", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0
    for res in results:
        assert isinstance(res, RetrievedEvidence)
        assert -1.0 <= res.similarity_score <= 1.0
        assert len(res.customer_message) > 0
        assert len(res.brand_reply) > 0
        assert len(res.conversation_id) > 0


def test_retriever_similarity_ordering():
    retriever = HistoricalRetriever()
    results = retriever.retrieve("I want to return an item", top_k=5)
    # Check that results are sorted in descending order of similarity
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True)
