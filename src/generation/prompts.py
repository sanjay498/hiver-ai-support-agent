"""
Prompt Templates for Grounded Customer Support Reply Generation.
Enforces strict anti-hallucination guardrails and Twitter support tone.
"""

from typing import List, Optional
from src.retrieval.retriever import RetrievedEvidence

SYSTEM_PROMPT = """You are an official, highly professional customer support agent on Twitter for Amazon (@AmazonHelp).
Your goal is to provide helpful, polite, concise, and grounded replies to customers.

STRICT OPERATIONAL RULES:
1. Grounding: Rely strictly on the provided historical brand resolutions as evidence of procedure and style. Never treat historical facts (order IDs, specific dates, refunds granted to past users) as facts about the current customer.
2. Anti-Hallucination: DO NOT invent policies, fake refund dates, fake tracking numbers, promotional discounts, or internal actions that have not occurred.
3. Missing Information: If resolving the issue requires account-specific details (e.g. order number, tracking ID), politely ask the customer for them, or advise them to reach out via secure direct message (DM) or 'Your Orders'.
4. Format & Brevity: Keep the response concise and under 280 characters suitable for a Twitter support tweet. Include a polite agent sign-off (e.g., ^CS or -Alex) typical of social support.
5. Privacy & Safety: Never ask for passwords, full credit card numbers, or sensitive financial tokens in public.
6. Clean Output: Never expose internal reasoning, instructions, or chain-of-thought in your response.

Return your response strictly in valid JSON matching this schema:
{
  "reply": "<The exact customer-facing tweet text>",
  "grounding_confidence": <float between 0.0 and 1.0 representing how well the answer is grounded in the retrieved precedents>,
  "used_evidence": [<list of integer evidence_ids that directly informed your response>]
}
"""


def format_generation_prompt(
    customer_message: str,
    predicted_intent: str,
    retrieved_evidence: List[RetrievedEvidence],
    context: Optional[str] = None,
) -> str:
    """Constructs the prompt for the grounded response generator."""
    sections = []
    
    sections.append(f"### Predicted Intent:\n{predicted_intent}")
    
    if context:
        sections.append(f"### Preceding Conversation Context:\n{context}")
        
    sections.append(f"### Current Customer Message:\n{customer_message}")
    
    sections.append("### Retrieved Historical Brand Resolutions (Precedents):")
    if not retrieved_evidence:
        sections.append("No historical precedent found.")
    else:
        for ev in retrieved_evidence:
            sections.append(
                f"[Evidence #{ev.evidence_id} | Similarity: {ev.similarity_score:.2f}]\n"
                f"Historical Customer: \"{ev.customer_message}\"\n"
                f"Amazon Agent Resolution: \"{ev.brand_reply}\"\n"
            )
            
    sections.append("Provide your JSON response:")
    return "\n\n".join(sections)
