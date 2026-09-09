"""Tests for Data Leakage Prevention."""

import json
from pathlib import Path
import pytest
from src.retrieval.retriever import HistoricalRetriever

GOLDEN_PATH = Path("data/golden/golden_set.json")


def test_golden_conversations_not_in_index():
    assert GOLDEN_PATH.exists(), "Golden evaluation dataset must exist"
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    golden_ids = {ex["conversation_id"] for ex in golden_data if ex.get("conversation_id")}
    assert len(golden_ids) > 0
    
    retriever = HistoricalRetriever()
    indexed_ids = {meta["conversation_id"] for meta in retriever.metadata}
    
    # Strict assertion: intersection between golden evaluation IDs and indexed IDs must be empty!
    overlap = golden_ids & indexed_ids
    assert len(overlap) == 0, f"LEAKAGE DETECTED! Found {len(overlap)} golden conversation IDs inside retrieval index: {list(overlap)[:5]}"


def test_query_conversation_id_filtered():
    retriever = HistoricalRetriever()
    # Pick an existing indexed ID
    if retriever.metadata:
        sample_meta = retriever.metadata[0]
        sample_cid = sample_meta["conversation_id"]
        sample_text = sample_meta["customer_message"]
        
        # Searching for the exact text but supplying its conversation_id as query_conversation_id
        results = retriever.retrieve(query=sample_text, top_k=5, query_conversation_id=sample_cid)
        retrieved_cids = [r.conversation_id for r in results]
        
        assert sample_cid not in retrieved_cids, "Query conversation ID was not excluded from retrieval results!"
