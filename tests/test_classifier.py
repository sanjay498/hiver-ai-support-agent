"""Tests for Intent Classifier & Baselines."""

import pytest
from src.intent.classifier import AIIntentClassifier, IntentPrediction
from src.intent.baseline import MajorityClassBaseline, TfidfLogisticRegressionBaseline


def test_intent_prediction_schema():
    pred = IntentPrediction(
        intent="delivery_status",
        confidence=0.88,
        reason="Customer is asking for delivery tracking."
    )
    assert pred.intent == "delivery_status"
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.reason) > 5


def test_majority_baseline():
    train_texts = ["where is package", "tracking order", "refund please"]
    train_labels = ["delivery_status", "delivery_status", "refund_status"]
    
    model = MajorityClassBaseline().fit(train_texts, train_labels)
    preds = model.predict(["anything else", "what time"])
    assert preds == ["delivery_status", "delivery_status"]


def test_tfidf_logistic_regression():
    train_texts = [
        "where is my package tracking",
        "where is my delivery status",
        "i want a refund for my return",
        "process my refund please"
    ]
    train_labels = ["delivery_status", "delivery_status", "refund_status", "refund_status"]
    
    model = TfidfLogisticRegressionBaseline().fit(train_texts, train_labels)
    pred_intent, conf = model.predict_with_confidence("where is my package")
    assert pred_intent == "delivery_status"
    assert 0.0 <= conf <= 1.0


def test_ai_intent_classifier_offline():
    classifier = AIIntentClassifier()
    pred = classifier.predict("Where is my package? The tracking link is not working.")
    assert isinstance(pred, IntentPrediction)
    assert pred.intent in classifier.valid_intents
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.reason) > 5
