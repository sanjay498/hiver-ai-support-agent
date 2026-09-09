"""
Main AI Intent Classifier.
Uses an LLM (OpenAI-compatible) with strict JSON schema validation,
or a calibrated embedding-based semantic fallback when running offline without API credentials.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("src.intent.classifier")

INTENTS_FILE = Path("data/processed/intents.json")


class IntentPrediction(BaseModel):
    intent: str = Field(..., description="The predicted intent label from the taxonomy.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")
    reason: str = Field(..., min_length=5, description="Explicit rationale for the intent classification.")


class AIIntentClassifier:
    """
    Production-grade Intent Classifier.
    Employs OpenAI-compatible LLM endpoint with prompt-grounded intent definitions,
    with an automatic semantic fallback when API keys are absent.
    """
    def __init__(
        self,
        taxonomy_path: Path = INTENTS_FILE,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Path = Path(".cache/intent_cache")
    ):
        self.taxonomy_path = taxonomy_path
        self.taxonomy = self._load_taxonomy()
        self.valid_intents = list(self.taxonomy.keys())
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, IntentPrediction] = {}
        
        # Lazy initialization of semantic fallback model
        self._semantic_embedder = None
        self._intent_embeddings = None

    def _load_taxonomy(self) -> Dict[str, Any]:
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Intents taxonomy file not found: {self.taxonomy_path}. Run intent discovery first.")
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_cache_key(self, message: str, context: Optional[str] = None) -> str:
        raw = f"{message.strip()}|||{context or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _check_disk_cache(self, cache_key: str) -> Optional[IntentPrediction]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return IntentPrediction(**data)
            except Exception:
                pass
        return None

    def _save_disk_cache(self, cache_key: str, prediction: IntentPrediction):
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(prediction.model_dump(), f)
        except Exception as e:
            logger.debug("Failed writing intent cache: %s", e)

    def _build_prompt(self, message: str, context: Optional[str] = None) -> str:
        prompt = [
            "You are an expert customer support intent classification engine for Amazon.",
            "Classify the following incoming customer message into EXACTLY ONE intent from the taxonomy below.\n",
            "### Intent Taxonomy & Definitions:"
        ]
        for intent_name, details in self.taxonomy.items():
            prompt.append(f"- **{intent_name}**: {details['description']}")
            if details.get("positive_examples"):
                ex_str = " | ".join([f'"{ex[:80]}"' for ex in details["positive_examples"][:2]])
                prompt.append(f"  Exemplars: {ex_str}")
                
        prompt.append("\n### Instructions:")
        prompt.append("1. Select the single best matching intent from the list above.")
        prompt.append("2. Output confidence score between 0.0 and 1.0.")
        prompt.append("3. Provide a concise, clear reason justifying the decision.")
        prompt.append("4. Return output strictly as a JSON object matching this schema:")
        prompt.append('{"intent": "<intent_name>", "confidence": <float>, "reason": "<explanation>"}')
        
        if context:
            prompt.append(f"\n### Conversation Context:\n{context}")
            
        prompt.append(f"\n### Customer Message:\n{message}\n")
        prompt.append("JSON response:")
        return "\n".join(prompt)

    def _predict_with_llm(self, message: str, context: Optional[str] = None) -> Optional[IntentPrediction]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=15.0)
            prompt = self._build_prompt(message, context)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a specialized customer support intent classifier. Respond only in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content.strip()
            data = json.loads(raw_content)
            
            # Map or validate intent
            pred_intent = data.get("intent", "").strip().lower()
            if pred_intent not in self.valid_intents:
                # Find closest intent or default
                pred_intent = "general_feedback_complaint"
                
            pred = IntentPrediction(
                intent=pred_intent,
                confidence=float(data.get("confidence", 0.85)),
                reason=str(data.get("reason", "Customer inquiry corresponds to this intent."))
            )
            return pred
        except Exception as e:
            logger.warning("LLM intent classification failed or timed out: %s. Using semantic fallback.", e)
            return None

    def _init_semantic_engine(self):
        if self._semantic_embedder is not None:
            return
        logger.info("Initializing offline semantic intent classifier fallback...")
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Set cache dir to avoid mac permissions issue
        os.environ["HF_HOME"] = str(Path(".cache/huggingface").absolute())
        from src.retrieval.embeddings import resolve_local_model_path
        load_path = resolve_local_model_path("all-MiniLM-L6-v2")
        self._semantic_embedder = SentenceTransformer(load_path)
        
        # Build reference representations for each intent
        intent_texts = []
        for intent_name, details in self.taxonomy.items():
            desc = details["description"]
            examples = " ".join(details.get("positive_examples", [])[:3])
            combined = f"{intent_name.replace('_', ' ')}: {desc}. Examples: {examples}"
            intent_texts.append(combined)
            
        embeddings = self._semantic_embedder.encode(intent_texts, normalize_embeddings=True)
        self._intent_embeddings = embeddings

    def _predict_with_semantic_fallback(self, message: str, context: Optional[str] = None) -> IntentPrediction:
        self._init_semantic_engine()
        import numpy as np
        
        query_text = f"{context} {message}".strip() if context else message
        query_emb = self._semantic_embedder.encode([query_text], normalize_embeddings=True)[0]
        
        similarities = np.dot(self._intent_embeddings, query_emb)
        best_idx = int(np.argmax(similarities))
        raw_sim = float(similarities[best_idx])
        
        # Calibrate similarity to 0.5 - 0.99 confidence
        confidence = round(min(max(raw_sim * 1.15, 0.45), 0.98), 2)
        best_intent = self.valid_intents[best_idx]
        
        reason = f"Customer message matches historical pattern for '{best_intent.replace('_', ' ')}' with semantic similarity score {raw_sim:.2f}."
        return IntentPrediction(intent=best_intent, confidence=confidence, reason=reason)

    def predict(self, message: str, context: Optional[str] = None) -> IntentPrediction:
        """Predict intent with caching, LLM execution, and fallback safety."""
        cache_key = self._get_cache_key(message, context)
        
        # 1. Memory Cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
            
        # 2. Disk Cache
        disk_cached = self._check_disk_cache(cache_key)
        if disk_cached:
            self._memory_cache[cache_key] = disk_cached
            return disk_cached
            
        # 3. LLM classification if API key is provided
        pred = None
        if self.api_key and len(self.api_key.strip()) > 5:
            pred = self._predict_with_llm(message, context)
            
        # 4. Semantic fallback if LLM is unavailable or offline
        if pred is None:
            pred = self._predict_with_semantic_fallback(message, context)
            
        # Save cache
        self._memory_cache[cache_key] = pred
        self._save_disk_cache(cache_key, pred)
        return pred
