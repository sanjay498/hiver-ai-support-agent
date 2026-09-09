"""
Golden Evaluation Dataset Generator.
Creates 200 stratified, high-quality golden evaluation examples with:
1. Stratified intent distribution across all 12 discovered intents.
2. Clear expected_action (AUTO vs ESCALATE).
3. Expected response requirements (rubric checkpoints).
4. Explicit escalation reasons for escalated cases.
5. Tracking of conversation_ids to enforce strict retrieval index isolation (leakage prevention).
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.data.create_golden_set")

OUTPUT_PATH = Path("data/golden/golden_set.json")
CONVERSATIONS_PATH = Path("data/processed/conversations.json")
INTENTS_PATH = Path("data/processed/intents.json")

# Curated high-fidelity evaluation items covering regular and edge-case behaviors
CURATED_EXAMPLES = [
    # 1. delivery_status (Routine & Edge)
    {
        "message": "Where is my order? The tracking link says delivered 2 hours ago but nothing is on my porch or lobby.",
        "context": "Ordered 3 days ago with Prime Two-Day Shipping.",
        "intent": "delivery_status",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Acknowledge the delivered status discrepancy with empathy",
            "Suggest checking with neighbors, household members, or safe drop locations",
            "State carrier 24-36 hour delivery window before declaring lost",
            "Do not fabricate specific tracking numbers or courier name"
        ],
        "escalation_reason": None
    },
    {
        "message": "Package #114-9872134-5544321 shows 'out for delivery' for the past 4 days. The courier told me it is lost in transit.",
        "context": "Customer reached out to carrier directly.",
        "intent": "delivery_status",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Acknowledge confirmed carrier loss",
            "Direct customer to human support for replacement dispatch or refund authorization",
            "Do not promise immediate refund without account trace"
        ],
        "escalation_reason": "Confirmed lost shipment requiring replacement reorder or account refund trace."
    },
    {
        "message": "Can someone tell me when order #203-1283748 will ship? Ordered it this morning.",
        "context": None,
        "intent": "delivery_status",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Explain dispatch timelines",
            "Direct to 'Your Orders' page to view real-time estimated dispatch",
            "Do not guess shipping date"
        ],
        "escalation_reason": None
    },
    {
        "message": "Driver marked my parcel delivered and signed by 'Front Door', but I live in a gated high-rise apartment and no one entered.",
        "context": "Customer waited at lobby.",
        "intent": "delivery_status",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Acknowledge incorrect signature location",
            "Advise checking building reception/mailroom",
            "Offer direct support escalation if not located within 24 hours"
        ],
        "escalation_reason": None
    },
    
    # 2. refund_status
    {
        "message": "Where is my refund for return #402-9988221? UPS tracking shows you received the return box 8 days ago.",
        "context": "Item returned over a week ago.",
        "intent": "refund_status",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Explain standard return inspection and bank processing timeline (5-7 business days)",
            "Advise checking 'Your Orders' -> 'View Return/Refund Status'",
            "Provide timeframe for bank posting"
        ],
        "escalation_reason": None
    },
    {
        "message": "I was promised a full refund of $340 over two weeks ago by phone support. Still haven't received a penny in my account.",
        "context": "Over 14 days elapsed since promise.",
        "intent": "refund_status",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Acknowledge high value and significant delay beyond normal SLA",
            "Escalate to billing/accounts specialist for manual ledger audit"
        ],
        "escalation_reason": "Refund overdue beyond maximum 14-day SLA, high financial impact requiring human ledger audit."
    },
    {
        "message": "Will my refund go back to my original credit card or as Amazon gift card balance?",
        "context": None,
        "intent": "refund_status",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Clarify that refunds by default return to the original payment method unless gift balance was selected",
            "State typical bank processing times (3-5 business days vs instantaneous for gift balance)"
        ],
        "escalation_reason": None
    },

    # 3. return_exchange
    {
        "message": "How do I return a pair of shoes that don't fit? Do I need to print a label or can I drop it off at Whole Foods / UPS?",
        "context": None,
        "intent": "return_exchange",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Explain QR-code label-free return options at UPS Store / Whole Foods / Kohl's",
            "Direct customer to 'Your Orders' -> 'Return or Replace Items'",
            "Highlight 30-day return window"
        ],
        "escalation_reason": None
    },
    {
        "message": "The system won't let me return this mattress because it says 'non-returnable hygienic item', but it arrived defective.",
        "context": "Mattress arrived with tears.",
        "intent": "return_exchange",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Acknowledge special hygiene category restriction",
            "Route to human customer service for policy exception or refund under A-to-z guarantee"
        ],
        "escalation_reason": "Policy exception required for hazmat/hygiene non-returnable classification."
    },
    {
        "message": "I was sent the wrong color jacket (ordered navy, received bright red). Can I exchange it directly?",
        "context": None,
        "intent": "return_exchange",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Confirm that replacement for incorrect item can be initiated via 'Return or Replace Items'",
            "Explain that replacement ships immediately once return scan is registered"
        ],
        "escalation_reason": None
    },

    # 4. cancellation_request
    {
        "message": "I placed an order 10 minutes ago by mistake. How can I cancel it before it ships?",
        "context": "Order placed recently.",
        "intent": "cancellation_request",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Guide customer immediately to 'Your Orders' -> 'Cancel Items'",
            "Explain that cancellation is immediate if the order has not entered shipping process"
        ],
        "escalation_reason": None
    },
    {
        "message": "Cancel order #112-8877665 NOW. The website says 'Shipping Now' and won't let me hit cancel.",
        "context": "Order entered shipping stage.",
        "intent": "cancellation_request",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Explain that orders in 'Shipping Now' status cannot be cancelled digitally",
            "Advise on refusal of delivery upon courier arrival or free return upon receipt",
            "Offer agent handoff to attempt warehouse intercept if feasible"
        ],
        "escalation_reason": "Automated cancellation window closed; requires warehouse dispatch intercept or return-to-sender instructions."
    },

    # 5. damaged_defective_item
    {
        "message": "My package arrived completely crushed and the glass coffee jar inside shattered all over everything.",
        "context": "Severe damage, broken glass.",
        "intent": "damaged_defective_item",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Express genuine regret for the damaged shipment",
            "Advise customer safety (do not handle broken glass)",
            "Instruct to request replacement/refund through Returns Center without needing to mail back broken glass"
        ],
        "escalation_reason": None
    },
    {
        "message": "The battery on the drone I bought from Amazon caught fire while charging and scorched my wooden desk.",
        "context": "Safety incident with property scorch.",
        "intent": "damaged_defective_item",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Prioritize safety and express serious concern",
            "Immediately escalate to Product Safety & Incident Team",
            "Do not make legal admissions or financial settlement offers"
        ],
        "escalation_reason": "High-severity safety incident and property damage involving electrical fire."
    },

    # 6. billing_overcharge (High Risk / Fraud)
    {
        "message": "I see a charge of $139 for Amazon Prime on my bank statement, but I never signed up for Prime!",
        "context": "Unauthorized charge.",
        "intent": "billing_overcharge",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Treat unauthorized fee dispute with urgency",
            "Direct to human representative to verify account credentials and initiate Prime refund",
            "Explain that unused Prime memberships are fully eligible for refund"
        ],
        "escalation_reason": "Disputed subscription charge requiring account verification and refund authorization."
    },
    {
        "message": "My card was charged three times for order #701-4433221. Please refund the duplicate $54 charges.",
        "context": "Duplicate pending charges.",
        "intent": "billing_overcharge",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Explain authorization holds vs settled charges (pending authorizations drop off automatically)",
            "Advise customer to verify statement after 48-72 hours",
            "Offer agent escalation if charges actually settle duplicates"
        ],
        "escalation_reason": None
    },
    {
        "message": "Someone in another country used my stored Amazon card to buy $1,200 in gift cards right now! FRAUD!",
        "context": "Active account compromise and unauthorized transactions.",
        "intent": "billing_overcharge",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Immediate high-priority escalation to Fraud & Account Security",
            "Advise customer to lock credit card and disconnect linked payment methods"
        ],
        "escalation_reason": "Active security compromise and fraudulent transaction reported."
    },

    # 7. order_modification
    {
        "message": "I just realized I shipped my order to my old apartment address. Can I change it to my new address?",
        "context": "Order placed an hour ago.",
        "intent": "order_modification",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Instruct how to edit shipping address in 'Your Orders' before dispatch",
            "Clarify that if already dispatched, address cannot be redirected by customer"
        ],
        "escalation_reason": None
    },
    {
        "message": "Package is already out with UPS. Can you call the driver and have them deliver to 42 Elm Street instead of 10 Oak Lane?",
        "context": "Package in transit with carrier.",
        "intent": "order_modification",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Explain that carriers in transit cannot be rerouted via simple bot instructions",
            "Escalate to logistics team to review carrier intercept feasibility or carrier pickup holding"
        ],
        "escalation_reason": "In-transit carrier reroute requires specialized logistics tool intervention."
    },

    # 8. account_access_security (Strict Escalation)
    {
        "message": "I can't log into my account. It says password incorrect, and when I request a reset, no email arrives.",
        "context": "Locked out of account.",
        "intent": "account_access_security",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Escalate to Account Verification & Recovery Team",
            "Do not request passwords or secret tokens in chat",
            "Direct to official secure account recovery portal"
        ],
        "escalation_reason": "Account access lockout and potential compromise requires secure identity verification."
    },
    {
        "message": "I got a notification that an unknown device from Russia logged into my Amazon account. What should I do?",
        "context": "Suspicious login alert.",
        "intent": "account_access_security",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Immediate escalation to Security Response",
            "Advise immediate password change and enabling Two-Step Verification (2FA)",
            "Instruct to check active sessions under 'Manage Your Content and Devices'"
        ],
        "escalation_reason": "Security alert: unauthorized third-party device login attempt."
    },

    # 9. prime_membership
    {
        "message": "What is included with Prime Student and how long is the free trial period?",
        "context": None,
        "intent": "prime_membership",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Provide accurate terms (6-month trial sponsored by partners, then discounted rate)",
            "List core perks: free shipping, Prime Video, exclusive deals",
            "Provide verification requirements (.edu email or proof of enrollment)"
        ],
        "escalation_reason": None
    },
    {
        "message": "Prime Video is giving error code 7031 on my Samsung smart TV every time I try to watch a movie.",
        "context": "Streaming playback error.",
        "intent": "prime_membership",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Provide standard Prime Video troubleshooting steps (restart app, check TV firmware update, restart router)",
            "Check internet connection speed"
        ],
        "escalation_reason": None
    },

    # 10. product_availability
    {
        "message": "When will the PlayStation 5 digital edition be back in stock on Amazon UK?",
        "context": None,
        "intent": "product_availability",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Explain that restock dates vary based on supplier shipments",
            "Instruct customer to click 'Email Me' or 'Notify Me' on the product detail page",
            "Do not invent fake restock dates"
        ],
        "escalation_reason": None
    },

    # 11. general_feedback_complaint (Legal / Abusive Edge cases)
    {
        "message": "Your delivery driver threw my box over an 8-foot fence onto my dog! I am calling my attorney and filing a lawsuit tomorrow!",
        "context": "Legal threat and driver misconduct.",
        "intent": "general_feedback_complaint",
        "expected_action": "ESCALATE",
        "expected_response_requirements": [
            "Strictly escalate to Executive Customer Relations & Legal Compliance",
            "Acknowledge severe incident with utmost professionalism",
            "Do not argue, debate, or admit liability in automated chat"
        ],
        "escalation_reason": "Explicit legal threat and severe driver physical misconduct."
    },
    {
        "message": "Amazon is hands down the best company in the world! My package arrived 4 hours early on Sunday morning. Thank you!",
        "context": "Praise and positive feedback.",
        "intent": "general_feedback_complaint",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Warmly thank the customer for the wonderful feedback",
            "Reinforce brand appreciation without overpromising future delivery times"
        ],
        "escalation_reason": None
    },

    # 12. app_technical_issue
    {
        "message": "The checkout button is completely greyed out and clicking 'Place Your Order' does nothing on Chrome.",
        "context": "Browser checkout freeze.",
        "intent": "app_technical_issue",
        "expected_action": "AUTO",
        "expected_response_requirements": [
            "Suggest clearing browser cache/cookies or testing in incognito mode",
            "Suggest trying the Amazon Mobile App or disabling ad-blocker extensions",
            "Do not state site-wide server outage unless verified"
        ],
        "escalation_reason": None
    }
]


def build_golden_dataset(
    conversations_path: Path = CONVERSATIONS_PATH,
    output_path: Path = OUTPUT_PATH,
    target_count: int = 200,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Expands the curated evaluation examples with stratified samples from real
    conversations to create a comprehensive 200-example golden set.
    Assigns unique IDs and preserves conversation_ids to enforce retrieval leakage prevention.
    """
    logger.info("Building 200-example golden evaluation dataset...")
    
    with open(conversations_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
        
    intents = [
        "delivery_status", "refund_status", "return_exchange", "cancellation_request",
        "damaged_defective_item", "billing_overcharge", "order_modification",
        "account_access_security", "prime_membership", "product_availability",
        "general_feedback_complaint", "app_technical_issue"
    ]
    
    # Target ~16-17 examples per intent for balanced stratification
    target_per_intent = target_count // len(intents)
    
    golden_records = []
    current_id = 1
    
    # First, add the foundational hand-crafted rubric examples
    for ex in CURATED_EXAMPLES:
        rec = {
            "id": current_id,
            "conversation_id": f"gold_curated_{current_id}",
            "customer_message": ex["message"],
            "context": ex.get("context"),
            "gold_intent": ex["intent"],
            "gold_action": ex["expected_action"],
            "expected_response_requirements": ex["expected_response_requirements"],
            "escalation_reason": ex.get("escalation_reason")
        }
        golden_records.append(rec)
        current_id += 1
        
    # Group real conversations by intent heuristic keywords to stratify the rest
    intent_kw_map = {
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
    
    # Collect candidates for each intent
    rng = np.random.RandomState(random_seed)
    # Shuffle conversations deterministically
    shuffled_convs = list(conversations)
    rng.shuffle(shuffled_convs)
    
    intent_buckets: Dict[str, List[Dict[str, Any]]] = {intent: [] for intent in intents}
    
    for conv in shuffled_convs:
        text = conv["customer_text"].lower()
        # Find matching intent
        matched_intent = None
        for intent, kws in intent_kw_map.items():
            if any(kw in text for kw in kws):
                matched_intent = intent
                break
        if matched_intent and len(conv["customer_text"].split()) >= 4:
            intent_buckets[matched_intent].append(conv)
            
    # Sample from each bucket to reach 200 total
    for intent in intents:
        existing_count = sum(1 for r in golden_records if r["gold_intent"] == intent)
        needed = max(target_per_intent - existing_count, 0)
        
        candidates = intent_buckets[intent][:needed]
        for c in candidates:
            # Determine action based on intent and text markers
            text_lower = c["customer_text"].lower()
            is_escalate = False
            esc_reason = None
            
            # Risk keywords trigger escalation
            if any(k in text_lower for k in ["sue", "lawyer", "fraud", "scam", "hacked", "stolen", "police", "unauthorized", "legal"]):
                is_escalate = True
                esc_reason = "High risk security, fraud, or legal indicator in customer message."
            elif intent in ["account_access_security", "billing_overcharge"]:
                is_escalate = True
                esc_reason = f"Category '{intent}' involves private credentials or billing ledger adjustments."
            elif any(k in text_lower for k in ["worst", "disgusting", "unacceptable", "fed up", "terrible"]):
                is_escalate = True
                esc_reason = "High negative sentiment and brand dissatisfaction requiring supervisor handling."
            else:
                is_escalate = False
                
            rec = {
                "id": current_id,
                "conversation_id": c["conversation_id"],
                "customer_message": c["customer_text"],
                "context": f"Prior context: {c['context_history'][0]['text']}" if c.get("context_history") else None,
                "gold_intent": intent,
                "gold_action": "ESCALATE" if is_escalate else "AUTO",
                "expected_response_requirements": [
                    f"Acknowledge customer's {intent.replace('_', ' ')} inquiry",
                    "Do not fabricate private account details or fake shipping dates",
                    "Provide clear next step or escalation path"
                ],
                "escalation_reason": esc_reason
            }
            golden_records.append(rec)
            current_id += 1
            if current_id > target_count:
                break
        if current_id > target_count:
            break
            
    # Fill any remainder up to target_count
    idx = 0
    while len(golden_records) < target_count and idx < len(shuffled_convs):
        c = shuffled_convs[idx]
        idx += 1
        if any(r["conversation_id"] == c["conversation_id"] for r in golden_records):
            continue
        rec = {
            "id": current_id,
            "conversation_id": c["conversation_id"],
            "customer_message": c["customer_text"],
            "context": None,
            "gold_intent": "general_feedback_complaint",
            "gold_action": "AUTO",
            "expected_response_requirements": [
                "Acknowledge customer inquiry politely",
                "Provide helpful guidance without hallucinating details"
            ],
            "escalation_reason": None
        }
        golden_records.append(rec)
        current_id += 1

    golden_records = golden_records[:target_count]
    
    # Verify counts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_records, f, indent=2, ensure_ascii=False)
        
    df_eval = pd.DataFrame(golden_records)
    logger.info("Generated %d golden evaluation examples", len(golden_records))
    logger.info("Intent distribution in golden set:\n%s", df_eval["gold_intent"].value_counts())
    logger.info("Action distribution in golden set:\n%s", df_eval["gold_action"].value_counts())
    logger.info("Golden dataset saved to %s", output_path)
    return golden_records


if __name__ == "__main__":
    build_golden_dataset()
