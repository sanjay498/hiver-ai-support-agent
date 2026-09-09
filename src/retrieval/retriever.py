"""
Historical Customer Support Resolution Retriever.
Uses FAISS IndexFlatIP for exact cosine similarity retrieval over historical resolutions.
Enforces strict leakage prevention by excluding golden evaluation conversations.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import faiss
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.retrieval.embeddings import TextEmbedder

logger = logging.getLogger("src.retrieval.retriever")

DEFAULT_INDEX_PATH = Path("data/processed/faiss_index.bin")
DEFAULT_METADATA_PATH = Path("data/processed/faiss_metadata.json")
DEFAULT_GOLDEN_PATH = Path("data/golden/golden_set.json")


class RetrievedEvidence(BaseModel):
    evidence_id: int = Field(..., description="1-indexed rank of retrieved evidence.")
    conversation_id: str = Field(..., description="Unique conversation ID in historical archive.")
    similarity_score: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score.")
    customer_message: str = Field(..., description="Historical customer message.")
    brand_reply: str = Field(..., description="Historical brand support agent resolution.")
    intent: Optional[str] = Field(None, description="Inferred or labeled intent of historical resolution.")


class HistoricalRetriever:
    """
    FAISS-based historical customer support retriever with strict leakage prevention.
    """
    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        index_path: Path = DEFAULT_INDEX_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        golden_path: Path = DEFAULT_GOLDEN_PATH
    ):
        self.embedder = embedder or TextEmbedder()
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.golden_path = Path(golden_path)
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        
        # Load golden IDs for leakage prevention
        self.excluded_conversation_ids: Set[str] = self._load_excluded_ids()
        
        # Attempt to load existing index if available
        if self.index_path.exists() and self.metadata_path.exists():
            self.load()

    def _load_excluded_ids(self) -> Set[str]:
        excluded = set()
        if self.golden_path.exists():
            try:
                with open(self.golden_path, "r", encoding="utf-8") as f:
                    golden_data = json.load(f)
                    for item in golden_data:
                        cid = item.get("conversation_id")
                        if cid:
                            excluded.add(cid)
                logger.info("Loaded %d conversation IDs to exclude from retrieval (leakage prevention)", len(excluded))
            except Exception as e:
                logger.warning("Could not read golden IDs for exclusion: %s", e)
        return excluded

    def build_index(
        self,
        conversations: List[Dict[str, Any]],
        save: bool = True
    ):
        """
        Builds FAISS index from conversation list, strictly skipping any golden evaluation IDs.
        """
        logger.info("Building FAISS retrieval index from %d candidate conversations...", len(conversations))
        
        filtered_convs = []
        skipped_for_leakage = 0
        seen_texts = set()
        
        for c in conversations:
            cid = c.get("conversation_id")
            if cid in self.excluded_conversation_ids:
                skipped_for_leakage += 1
                continue
            
            cust_text = c.get("customer_text", "").strip()
            brand_reply = c.get("brand_reply", "").strip()
            
            if not cust_text or not brand_reply or len(cust_text.split()) < 3:
                continue
                
            norm_key = cust_text.lower()
            if norm_key in seen_texts:
                continue
            seen_texts.add(norm_key)
            
            filtered_convs.append(c)
            
        logger.info("Filtered %d indexable conversations (Skipped %d golden IDs for leakage prevention)",
                    len(filtered_convs), skipped_for_leakage)
                    
        if not filtered_convs:
            raise ValueError("No conversations remaining to index after leakage filtering!")
            
        texts_to_embed = [c["customer_text"] for c in filtered_convs]
        embeddings = self.embedder.embed_texts(texts_to_embed)
        
        # Build FAISS IndexFlatIP (exact cosine similarity on normalized vectors)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        self.metadata = []
        for idx, c in enumerate(filtered_convs):
            self.metadata.append({
                "idx": idx,
                "conversation_id": c["conversation_id"],
                "customer_message": c["customer_text"],
                "brand_reply": c["brand_reply"],
                "context_history": c.get("context_history", []),
                "thread_length": c.get("thread_length", 2)
            })
            
        logger.info("FAISS index built with %d entries (dimension %d)", self.index.ntotal, dimension)
        if save:
            self.save()

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info("Saved FAISS index to %s and metadata to %s", self.index_path, self.metadata_path)

    def load(self):
        logger.info("Loading FAISS index from %s...", self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info("Loaded index with %d items and %d metadata records", self.index.ntotal, len(self.metadata))

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        query_conversation_id: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[RetrievedEvidence]:
        """
        Retrieves top_k similar historical resolutions, filtering out any leakage.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Retriever index is not initialized or empty. Call build_index() first.")
            
        # Embed query
        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        
        # Fetch slightly more to allow post-filtering
        fetch_k = min(top_k * 3 + 5, self.index.ntotal)
        scores, indices = self.index.search(query_vec, fetch_k)
        
        results: List[RetrievedEvidence] = []
        rank = 1
        
        for sim, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            cid = meta["conversation_id"]
            
            # Strict leakage check
            if query_conversation_id and cid == query_conversation_id:
                continue
            if cid in self.excluded_conversation_ids:
                continue
            if float(sim) < min_similarity:
                continue
                
            results.append(RetrievedEvidence(
                evidence_id=rank,
                conversation_id=cid,
                similarity_score=round(float(sim), 4),
                customer_message=meta["customer_message"],
                brand_reply=meta["brand_reply"],
                intent=meta.get("intent")
            ))
            rank += 1
            if len(results) >= top_k:
                break
                
        return results
