"""
Escalation Policy & Decision Engine.
Implements conservative, safety-critical decision logic to route inquiries
between AUTO_HANDLE and ESCALATE based on:
1. Critical Safety & Risk Keywords (Legal, Fraud, Abuse, Security, Safety Hazards).
2. Category-Specific Policy Matrix (e.g. Account Security and Billing Disputes require human ledger access).
3. Classifier Confidence Gates (configurable threshold, default 0.65-0.70).
4. Retrieval Evidence & Precedent Thresholds (configurable similarity threshold, default 0.55-0.58).
5. Grounding Confidence Gates.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.retrieval.retriever import RetrievedEvidence

logger = logging.getLogger("src.escalation.policy")


class EscalationDecision(BaseModel):
    decision: str = Field(..., description="'AUTO_HANDLE' or 'ESCALATE'")
    reason: str = Field(..., description="Explicit operational rationale for the routing decision.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence score.")
    risk_flags: List[str] = Field(default_factory=list, description="Detected safety or policy risk triggers.")


# Strict risk dictionaries
RISK_PATTERNS = {
    "LEGAL_THREAT": [
        r"\blawyer\b", r"\battorney\b", r"\bsue\b", r"\bsuing\b", r"\blawsuit\b",
        r"\blegal action\b", r"\bcourt\b", r"\bftc\b", r"\bconsumer court\b", r"\bbetter business bureau\b", r"\bbbb\b"
    ],
    "FRAUD_SECURITY_COMPROMISE": [
        r"\bhacked\b", r"\bunauthorized\b", r"\bfraud\b", r"\bscam\b", r"\bscammed\b", r"\bstolen card\b",
        r"\bstolen account\b", r"\bidentity theft\b", r"\bcompromised\b", r"\bunauthorized charge\b"
    ],
    "SAFETY_HAZARD": [
        r"\bfire\b", r"\bexplosion\b", r"\bexplode\b", r"\binjury\b", r"\bhospital\b",
        r"\bburn\b", r"\belectrocuted\b", r"\bpoison\b", r"\bhazard\b", r"\bdangerous\b"
    ],
    "SEVERE_ABUSE_SENTIMENT": [
        r"\bfuck\b", r"\bshit\b", r"\basshole\b", r"\bbastard\b", r"\bdisgusting\b",
        r"\bunacceptable\b", r"\bworst company\b", r"\bpolice\b"
    ]
}

# Intents requiring human verification by default
MANDATORY_ESCALATION_INTENTS = {
    "account_access_security": "Account access and credential recovery require secure human verification.",
    "billing_overcharge": "Payment disputes and ledger adjustments require authorized financial specialist review."
}

# Defaults
CONFIDENCE_THRESHOLD = 0.65
RETRIEVAL_SIMILARITY_THRESHOLD = 0.55
GROUNDING_THRESHOLD = 0.60


class EscalationEngine:
    """
    Evaluates customer requests against conservative escalation rules.
    Favors human safety over risky automation.
    """
    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        retrieval_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
        grounding_threshold: float = GROUNDING_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold
        self.grounding_threshold = grounding_threshold

    def _scan_risk_keywords(self, text: str) -> List[str]:
        flags = []
        text_lower = text.lower()
        for category, patterns in RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    flags.append(f"{category}:{pattern.replace(r'\b', '')}")
                    break
        return flags

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieved_evidence: List[RetrievedEvidence],
        grounding_confidence: float = 0.85,
        context: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        retrieval_threshold: Optional[float] = None,
    ) -> EscalationDecision:
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        ret_thresh = retrieval_threshold if retrieval_threshold is not None else self.retrieval_threshold
        
        combined_text = f"{context} {customer_message}".strip() if context else customer_message
        risk_flags = self._scan_risk_keywords(combined_text)
        
        # 1. Critical Risk Check (Legal, Fraud, Safety, Severe Abuse)
        if risk_flags:
            flag_names = ", ".join([f.split(":")[0] for f in risk_flags])
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Immediate escalation triggered by risk policy flags: {flag_names}.",
                confidence=0.98,
                risk_flags=risk_flags
            )

        # 2. Mandatory Intent Policy Matrix Check
        if predicted_intent in MANDATORY_ESCALATION_INTENTS:
            policy_reason = MANDATORY_ESCALATION_INTENTS[predicted_intent]
            return EscalationDecision(
                decision="ESCALATE",
                reason=policy_reason,
                confidence=0.95,
                risk_flags=[f"POLICY_INTENT:{predicted_intent}"]
            )

        # 3. Classifier Confidence Gate
        if intent_confidence < conf_thresh:
            return EscalationDecision(
                decision="ESCALATE",
                reason=(
                    f"Low intent classification confidence ({intent_confidence:.2f} < "
                    f"{conf_thresh:.2f}). Escalating to prevent misunderstanding."
                ),
                confidence=0.88,
                risk_flags=["LOW_INTENT_CONFIDENCE"]
            )

        # 4. Retrieval Precedent Gate
        if not retrieved_evidence:
            return EscalationDecision(
                decision="ESCALATE",
                reason="No historical resolution precedent found in knowledge base.",
                confidence=0.90,
                risk_flags=["NO_RETRIEVAL_PRECEDENT"]
            )
            
        top_sim = retrieved_evidence[0].similarity_score
        if top_sim < ret_thresh:
            return EscalationDecision(
                decision="ESCALATE",
                reason=(
                    f"Insufficient historical precedent similarity ({top_sim:.2f} < "
                    f"{ret_thresh:.2f}). Risk of hallucinated or irrelevant guidance."
                ),
                confidence=0.85,
                risk_flags=["LOW_RETRIEVAL_SIMILARITY"]
            )

        # 5. Grounding Confidence Gate
        if grounding_confidence < self.grounding_threshold:
            return EscalationDecision(
                decision="ESCALATE",
                reason=(
                    f"Draft reply grounding confidence ({grounding_confidence:.2f} < "
                    f"{self.grounding_threshold:.2f}) falls below safety threshold."
                ),
                confidence=0.84,
                risk_flags=["LOW_GROUNDING_CONFIDENCE"]
            )

        # 6. Eligible for Safe Auto-Handling
        return EscalationDecision(
            decision="AUTO_HANDLE",
            reason=(
                f"Routine {predicted_intent.replace('_', ' ')} inquiry with high intent confidence "
                f"({intent_confidence:.2f} >= {conf_thresh:.2f}) and verified historical precedent ({top_sim:.2f} >= {ret_thresh:.2f} similarity)."
            ),
            confidence=round(min((intent_confidence + top_sim) / 2.0, 0.96), 2),
            risk_flags=[]
        )
