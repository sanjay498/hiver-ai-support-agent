"""Tests for Escalation Policy Engine."""

import pytest
from src.escalation.policy import EscalationEngine, EscalationDecision
from src.retrieval.retriever import RetrievedEvidence


@pytest.fixture
def escalation_engine():
    return EscalationEngine()


@pytest.fixture
def mock_evidence():
    return [
        RetrievedEvidence(
            evidence_id=1,
            conversation_id="conv_test_1",
            similarity_score=0.76,
            customer_message="Where is my order?",
            brand_reply="Check Your Orders page ^CS"
        )
    ]


def test_safe_auto_handle(escalation_engine, mock_evidence):
    res = escalation_engine.evaluate(
        customer_message="Where is my package? The tracking link says delayed.",
        predicted_intent="delivery_status",
        intent_confidence=0.85,
        retrieved_evidence=mock_evidence,
        grounding_confidence=0.85
    )
    assert res.decision == "AUTO_HANDLE"
    assert "Routine delivery status inquiry" in res.reason
    assert len(res.risk_flags) == 0


def test_legal_threat_escalation(escalation_engine, mock_evidence):
    res = escalation_engine.evaluate(
        customer_message="I am hiring an attorney and suing Amazon for this fraud!",
        predicted_intent="delivery_status",
        intent_confidence=0.90,
        retrieved_evidence=mock_evidence,
        grounding_confidence=0.85
    )
    assert res.decision == "ESCALATE"
    assert any("LEGAL_THREAT" in flag for flag in res.risk_flags)


def test_account_security_escalation(escalation_engine, mock_evidence):
    res = escalation_engine.evaluate(
        customer_message="I cannot log into my account. Password was changed.",
        predicted_intent="account_access_security",
        intent_confidence=0.95,
        retrieved_evidence=mock_evidence,
        grounding_confidence=0.85
    )
    assert res.decision == "ESCALATE"
    assert any("POLICY_INTENT" in flag for flag in res.risk_flags)


def test_low_confidence_escalation(escalation_engine, mock_evidence):
    res = escalation_engine.evaluate(
        customer_message="What time do you close?",
        predicted_intent="general_feedback_complaint",
        intent_confidence=0.55,  # Below 0.70 threshold
        retrieved_evidence=mock_evidence,
        grounding_confidence=0.80
    )
    assert res.decision == "ESCALATE"
    assert "LOW_INTENT_CONFIDENCE" in res.risk_flags
