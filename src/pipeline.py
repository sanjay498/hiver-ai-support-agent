"""
End-to-End Customer Support Agent Pipeline.
Coordinates:
1. Intent Classification (AI Classifier with fallback)
2. Historical Resolution Retrieval (FAISS index with leakage prevention)
3. Grounded Response Generation (Anti-hallucination prompted LLM / grounded synthesis)
4. Safety-Critical Escalation Decision Engine
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from src.intent.classifier import AIIntentClassifier, IntentPrediction
from src.retrieval.retriever import HistoricalRetriever, RetrievedEvidence
from src.generation.response_generator import ResponseGenerator, GeneratedReply
from src.escalation.policy import EscalationEngine, EscalationDecision

logger = logging.getLogger("src.pipeline")


class SupportAgentResponse(BaseModel):
    customer_message: str
    context: Optional[str] = None
    predicted_intent: str
    intent_confidence: float
    intent_reason: str
    decision: str = Field(..., description="'AUTO_HANDLE' or 'ESCALATE'")
    escalation_reason: str
    escalation_confidence: float
    risk_flags: List[str]
    draft_reply: str
    grounding_confidence: float
    evidence: List[RetrievedEvidence]


class CustomerSupportPipeline:
    """
    Unified production pipeline for automated AI customer support.
    """
    def __init__(
        self,
        classifier: Optional[AIIntentClassifier] = None,
        retriever: Optional[HistoricalRetriever] = None,
        generator: Optional[ResponseGenerator] = None,
        escalation_engine: Optional[EscalationEngine] = None
    ):
        logger.info("Initializing CustomerSupportPipeline components...")
        self.classifier = classifier or AIIntentClassifier()
        self.retriever = retriever or HistoricalRetriever()
        self.generator = generator or ResponseGenerator()
        self.escalation_engine = escalation_engine or EscalationEngine()
        logger.info("CustomerSupportPipeline initialized successfully.")

    def process(
        self,
        customer_message: str,
        context: Optional[str] = None,
        query_conversation_id: Optional[str] = None,
        top_k_evidence: int = 3
    ) -> SupportAgentResponse:
        """
        Executes complete end-to-end processing of an incoming customer inquiry.
        """
        # 1. Intent Classification
        intent_pred: IntentPrediction = self.classifier.predict(customer_message, context)
        
        # 2. Historical Precedent Retrieval
        evidence: List[RetrievedEvidence] = self.retriever.retrieve(
            query=customer_message,
            top_k=top_k_evidence,
            query_conversation_id=query_conversation_id
        )
        
        # 3. Grounded Response Generation
        draft: GeneratedReply = self.generator.generate(
            customer_message=customer_message,
            predicted_intent=intent_pred.intent,
            evidence=evidence,
            context=context
        )
        
        # 4. Escalation Policy Evaluation
        escalation: EscalationDecision = self.escalation_engine.evaluate(
            customer_message=customer_message,
            predicted_intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            retrieved_evidence=evidence,
            grounding_confidence=draft.grounding_confidence,
            context=context
        )
        
        return SupportAgentResponse(
            customer_message=customer_message,
            context=context,
            predicted_intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            intent_reason=intent_pred.reason,
            decision=escalation.decision,
            escalation_reason=escalation.reason,
            escalation_confidence=escalation.confidence,
            risk_flags=escalation.risk_flags,
            draft_reply=draft.reply,
            grounding_confidence=draft.grounding_confidence,
            evidence=evidence
        )


if __name__ == "__main__":
    pipeline = CustomerSupportPipeline()
    test_msg = "My package has been delayed for 4 days, where is it?"
    res = pipeline.process(test_msg)
    print("\n--- Pipeline Test Output ---")
    print(json.dumps(res.model_dump(), indent=2))
