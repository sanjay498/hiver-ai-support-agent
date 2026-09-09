"""
Baseline Evaluation Module.
Evaluates:
1. Baseline A: Majority Class Classifier
2. Baseline B: TF-IDF + Logistic Regression Classifier
against the Golden Evaluation Set.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.intent.baseline import MajorityClassBaseline, TfidfLogisticRegressionBaseline

logger = logging.getLogger("src.evaluation.baselines")

DEFAULT_GOLDEN_PATH = Path("data/golden/golden_set.json")
DEFAULT_TAXONOMY_PATH = Path("data/processed/intents.json")
DEFAULT_CONV_PATH = Path("data/processed/conversations.json")


def train_and_eval_baselines(
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    conv_path: Path = DEFAULT_CONV_PATH
) -> Dict[str, Any]:
    """
    Trains Baseline A & B on historical support training examples and evaluates on golden set.
    """
    logger.info("Loading golden evaluation set from %s...", golden_path)
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    test_texts = [ex["customer_message"] for ex in golden_data]
    test_labels = [ex["gold_intent"] for ex in golden_data]
    
    # Prepare training corpus: positive exemplars from taxonomy + subset of historical conversations
    train_texts = []
    train_labels = []
    
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
        for intent_name, details in taxonomy.items():
            for ex in details.get("positive_examples", []):
                train_texts.append(ex)
                train_labels.append(intent_name)
                
    # Also add keyword-matched training conversations from non-golden pool
    with open(conv_path, "r", encoding="utf-8") as f:
        convs = json.load(f)
        
    golden_ids = {ex.get("conversation_id") for ex in golden_data}
    intent_kws = {
        "delivery_status": ["tracking", "deliver", "package", "where is", "courier", "late"],
        "refund_status": ["refund", "reimburse", "money back", "credit back"],
        "return_exchange": ["return", "exchange", "replace", "drop-off", "send back"],
        "cancellation_request": ["cancel", "cancellation", "stop order"],
        "damaged_defective_item": ["damaged", "broken", "crushed", "shattered", "leaked", "defective"],
        "billing_overcharge": ["charge", "billed", "bank", "overcharged", "card"],
        "order_modification": ["address", "change", "delivery date", "apartment"],
        "account_access_security": ["password", "login", "locked", "otp", "2fa", "security"],
        "prime_membership": ["prime", "membership", "video", "prime trial"],
        "product_availability": ["stock", "out of stock", "availability", "pre-order"],
        "general_feedback_complaint": ["service", "driver", "unacceptable", "terrible", "disappointed"],
        "app_technical_issue": ["app", "website", "glitch", "crash", "error", "button"]
    }
    
    for c in convs:
        if c["conversation_id"] in golden_ids:
            continue
        text_lower = c["customer_text"].lower()
        for intent, kws in intent_kws.items():
            if any(kw in text_lower for kw in kws):
                train_texts.append(c["customer_text"])
                train_labels.append(intent)
                break
                
    logger.info("Training set size for baselines: %d samples", len(train_texts))
    
    # 1. Baseline A: Majority Class
    maj_baseline = MajorityClassBaseline().fit(train_texts, train_labels)
    maj_preds = maj_baseline.predict(test_texts)
    maj_acc = accuracy_score(test_labels, maj_preds)
    maj_macro_f1 = f1_score(test_labels, maj_preds, average="macro", zero_division=0)
    maj_weighted_f1 = f1_score(test_labels, maj_preds, average="weighted", zero_division=0)
    
    # 2. Baseline B: TF-IDF + Logistic Regression
    lr_baseline = TfidfLogisticRegressionBaseline().fit(train_texts, train_labels)
    lr_preds = lr_baseline.predict(test_texts)
    lr_acc = accuracy_score(test_labels, lr_preds)
    lr_macro_f1 = f1_score(test_labels, lr_preds, average="macro", zero_division=0)
    lr_weighted_f1 = f1_score(test_labels, lr_preds, average="weighted", zero_division=0)
    
    results = {
        "baseline_majority": {
            "name": "Baseline A (Majority Class)",
            "accuracy": round(float(maj_acc), 4),
            "macro_f1": round(float(maj_macro_f1), 4),
            "weighted_f1": round(float(maj_weighted_f1), 4),
            "predicted_class": maj_baseline.majority_intent
        },
        "baseline_tfidf_lr": {
            "name": "Baseline B (TF-IDF + Logistic Regression)",
            "accuracy": round(float(lr_acc), 4),
            "macro_f1": round(float(lr_macro_f1), 4),
            "weighted_f1": round(float(lr_weighted_f1), 4)
        }
    }
    logger.info("Baseline Evaluation Complete:")
    logger.info("  Majority Baseline -> Acc: %.3f, Macro-F1: %.3f", maj_acc, maj_macro_f1)
    logger.info("  TF-IDF + LR Baseline -> Acc: %.3f, Macro-F1: %.3f", lr_acc, lr_macro_f1)
    return results


if __name__ == "__main__":
    res = train_and_eval_baselines()
    print(json.dumps(res, indent=2))
