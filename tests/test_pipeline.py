"""Tests for End-to-End Pipeline."""

import pytest
from src.pipeline import CustomerSupportPipeline, SupportAgentResponse


@pytest.fixture(scope="module")
def pipeline():
    return CustomerSupportPipeline()


def test_end_to_end_safe_routine_flow(pipeline):
    query = "Where is my package? Delayed for 2 days."
    resp = pipeline.process(query)
    
    assert isinstance(resp, SupportAgentResponse)
    assert resp.predicted_intent == "delivery_status"
    assert resp.decision in ["AUTO_HANDLE", "ESCALATE"]
    assert len(resp.draft_reply) > 10
    assert len(resp.evidence) > 0
    assert resp.intent_confidence > 0.0


def test_end_to_end_escalation_flow(pipeline):
    query = "I am going to sue Amazon in federal court for stolen money!"
    resp = pipeline.process(query)
    
    assert isinstance(resp, SupportAgentResponse)
    assert resp.decision == "ESCALATE"
    assert any("LEGAL_THREAT" in f for f in resp.risk_flags)
    assert "Immediate escalation" in resp.escalation_reason
