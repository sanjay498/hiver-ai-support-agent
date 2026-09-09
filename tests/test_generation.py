"""Tests for Grounded Response Generator."""

import pytest
from src.generation.response_generator import ResponseGenerator, GeneratedReply
from src.retrieval.retriever import RetrievedEvidence


def test_generated_reply_schema():
    reply = GeneratedReply(
        reply="Hello! You can track your order in Your Orders at amazon.com/orders. ^CS",
        grounding_confidence=0.88,
        used_evidence=[1]
    )
    assert len(reply.reply) > 10
    assert 0.0 <= reply.grounding_confidence <= 1.0
    assert reply.used_evidence == [1]


def test_generator_synthesis_offline():
    gen = ResponseGenerator()
    evidence = [
        RetrievedEvidence(
            evidence_id=1,
            conversation_id="conv_1",
            similarity_score=0.82,
            customer_message="Where is my package?",
            brand_reply="Please check Your Orders at amazon.com/orders ^CS"
        )
    ]
    result = gen.generate(
        customer_message="Where is my package? It is late.",
        predicted_intent="delivery_status",
        evidence=evidence
    )
    assert isinstance(result, GeneratedReply)
    assert len(result.reply) > 15
    assert len(result.reply) <= 280
    assert "^CS" in result.reply
    assert result.grounding_confidence >= 0.70
