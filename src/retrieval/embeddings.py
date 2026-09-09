"""
Embedding Generation Wrapper.
Uses sentence-transformers ('all-MiniLM-L6-v2') with L2 normalization
and automated local snapshot resolution for fast, offline operation.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("src.retrieval.embeddings")

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def resolve_local_model_path(model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Finds local snapshot path if available to ensure instant offline loading."""
    hub_path = Path(".cache/huggingface/hub").resolve()
    slug = model_name.replace("/", "--")
    if not slug.startswith("models--"):
        slug = f"models--{slug}"
    if not slug.startswith("models--sentence-transformers--") and "sentence-transformers" not in slug:
        slug = f"models--sentence-transformers--{model_name}"
        
    candidates = list(hub_path.glob(f"{slug}/snapshots/*"))
    if candidates and candidates[0].is_dir():
        local_path = str(candidates[0])
        logger.debug("Resolved local snapshot path: %s", local_path)
        return local_path
    return model_name


class TextEmbedder:
    """Singleton-style embedder wrapping SentenceTransformer."""
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        cache_dir = Path(".cache/huggingface").resolve()
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        load_path = resolve_local_model_path(model_name)
        logger.info("Initializing TextEmbedder using path: %s", load_path)
        self.model = SentenceTransformer(load_path)
        self.dimension = self.model.get_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Generates L2-normalized float32 embeddings for a list of strings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Generates L2-normalized float32 1D embedding for a single query string."""
        emb = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(emb, dtype=np.float32)
