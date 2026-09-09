"""
Intent Classification Baselines.
Implements:
1. Baseline A: Majority Class Classifier (Trivial baseline, always predicts most frequent intent).
2. Baseline B: TF-IDF + Logistic Regression Classifier (Standard ML baseline with sublinear TF-IDF).
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("src.intent.baseline")


class MajorityClassBaseline:
    """Predicts the single most frequent intent observed during training."""
    def __init__(self):
        self.majority_intent: str = "delivery_status"
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        counts = pd.Series(labels).value_counts()
        self.majority_intent = counts.index[0]
        self.classes_ = sorted(list(set(labels)))
        logger.info("MajorityClassBaseline trained: majority intent is '%s' (%d occurrences)",
                    self.majority_intent, counts.iloc[0])
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_intent] * len(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        n_samples = len(texts)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        maj_idx = self.classes_.index(self.majority_intent)
        proba[:, maj_idx] = 1.0
        return proba


class TfidfLogisticRegressionBaseline:
    """
    TF-IDF N-gram feature extraction + Calibrated Multi-Class Logistic Regression.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
            stop_words="english",
            min_df=2
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_seed
        )
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        logger.info("Training TfidfLogisticRegressionBaseline on %d samples...", len(texts))
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        logger.info("TfidfLogisticRegressionBaseline trained successfully with %d classes.", len(self.classes_))
        return self

    def predict(self, texts: List[str]) -> List[str]:
        X = self.vectorizer.transform(texts)
        return list(self.model.predict(X))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_intent = self.classes_[pred_idx]
        confidence = float(proba[pred_idx])
        return pred_intent, round(confidence, 3)
