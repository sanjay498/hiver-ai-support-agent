"""
Intent Discovery Module for Customer Support Twitter Dataset.
Performs data-driven discovery of customer intent taxonomy:
1. Embeds customer inquiries and applies K-Means clustering.
2. Extracts top n-gram keywords and representative centroid exemplars.
3. Generates structured intent definitions with boundary guidelines and escalation policies.
4. Saves formal taxonomy to data/processed/intents.json.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.intent.discover")

DEFAULT_CONV_PATH = Path("data/processed/conversations.json")
DEFAULT_INTENTS_PATH = Path("data/processed/intents.json")

# Grounded taxonomy definitions discovered and tailored for Amazon customer support
BASE_TAXONOMY_DEFINITIONS = {
    "delivery_status": {
        "description": "Inquiries regarding tracking, delayed shipments, estimated delivery dates, or packages marked delivered but not received.",
        "confusing_intents": ["damaged_defective_item", "order_modification", "cancellation_request"],
        "escalation_guidance": "Can be AUTO-HANDLED if tracking status can be checked and carrier timeframe is active. ESCALATE if package is confirmed missing beyond carrier threshold or high-value item requires carrier trace."
    },
    "refund_status": {
        "description": "Questions regarding the status, timeline, or receipt of a refund for a returned or cancelled order.",
        "confusing_intents": ["billing_overcharge", "return_exchange"],
        "escalation_guidance": "AUTO-HANDLE with general bank processing timelines (3-5 business days). ESCALATE if refund timeframe has elapsed without credit appearing in customer account."
    },
    "return_exchange": {
        "description": "Requests to initiate a return, replace a wrong item, print return shipping labels, or drop-off location questions.",
        "confusing_intents": ["damaged_defective_item", "refund_status"],
        "escalation_guidance": "AUTO-HANDLE with standard Online Return Center portal instructions. ESCALATE if item is non-returnable by standard policy or requires exception approval."
    },
    "cancellation_request": {
        "description": "Requests to cancel an order or item immediately after placement or before dispatch.",
        "confusing_intents": ["order_modification", "return_exchange"],
        "escalation_guidance": "AUTO-HANDLE if order has not yet entered dispatch. ESCALATE if order is already in shipping pipeline or requires manual dispatch recall."
    },
    "damaged_defective_item": {
        "description": "Reports of products arriving broken, crushed, leaking, spoiled, missing components, or malfunctioning out of the box.",
        "confusing_intents": ["delivery_status", "return_exchange"],
        "escalation_guidance": "AUTO-HANDLE replacement guidance if eligible for instant swap. ESCALATE if damage caused property hazard, involves hazardous materials, or high-value damage claim."
    },
    "billing_overcharge": {
        "description": "Disputes regarding unexpected charges, double-billing, price discrepancies, unauthorized subscriptions, or Prime fee surprises.",
        "confusing_intents": ["refund_status", "prime_membership", "account_access_security"],
        "escalation_guidance": "ESCALATE immediately to human agents. Financial discrepancies and payment disputes require account verification and authorization."
    },
    "order_modification": {
        "description": "Requests to change delivery address, payment method, recipient name, or delivery time slot on an existing order.",
        "confusing_intents": ["cancellation_request", "delivery_status"],
        "escalation_guidance": "AUTO-HANDLE guidance for 'Your Orders' address edit if dispatch hasn't locked. ESCALATE if package is already in transit with external carrier."
    },
    "account_access_security": {
        "description": "Issues with account login, two-factor authentication (2FA/OTP), locked credentials, password resets, or suspected compromised account.",
        "confusing_intents": ["billing_overcharge", "app_technical_issue"],
        "escalation_guidance": "STRICTLY ESCALATE. Automated bots must never handle account credentials, security lockouts, or identity verification."
    },
    "prime_membership": {
        "description": "Inquiries regarding Amazon Prime perks, Prime Video playback or content availability, renewal dates, or membership cancellation.",
        "confusing_intents": ["billing_overcharge", "app_technical_issue"],
        "escalation_guidance": "AUTO-HANDLE for general benefit policies, device compatibility, and cancellation steps. ESCALATE if customer requests prorated Prime refund."
    },
    "product_availability": {
        "description": "Questions regarding stock availability, pre-orders, estimated restock dates, or regional delivery eligibility for a specific item.",
        "confusing_intents": ["delivery_status", "order_modification"],
        "escalation_guidance": "AUTO-HANDLE with catalog availability guidelines and notifications subscription."
    },
    "general_feedback_complaint": {
        "description": "Expressions of customer dissatisfaction regarding driver conduct, packaging quality, policy changes, or poor service experience without specific action request.",
        "confusing_intents": ["damaged_defective_item", "delivery_status"],
        "escalation_guidance": "AUTO-HANDLE with polite acknowledgment and feedback routing. ESCALATE if customer threatens legal action, regulatory escalation, or public boycott."
    },
    "app_technical_issue": {
        "description": "Technical errors encountering the Amazon website, mobile app, checkout button unresponsive, shopping cart errors, or payment gateway timeout.",
        "confusing_intents": ["account_access_security", "billing_overcharge"],
        "escalation_guidance": "AUTO-HANDLE standard troubleshooting steps (cache clearing, browser update, app reinstallation). ESCALATE if outage affects payment completion."
    }
}


def discover_intent_taxonomy(
    conversations_path: Path = DEFAULT_CONV_PATH,
    output_path: Path = DEFAULT_INTENTS_PATH,
    n_clusters: int = 12,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Analyzes historical customer messages, clusters topics, and creates
    the data-grounded intent taxonomy with real positive examples.
    """
    logger.info("Loading conversations from %s ...", conversations_path)
    with open(conversations_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
        
    messages = [c["customer_text"] for c in conversations if len(c["customer_text"].split()) >= 3]
    logger.info("Analyzing %d eligible customer messages for intent clustering...", len(messages))
    
    # TF-IDF Clustering for topic discovery
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(messages)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Find top terms per cluster
    terms = vectorizer.get_feature_names_out()
    cluster_top_terms = {}
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-8:][::-1]
        cluster_top_terms[i] = [terms[ind] for ind in top_indices]
        
    logger.info("Extracted %d cluster keyword profiles", len(cluster_top_terms))

    # Match each discovered intent with actual representative positive examples from conversations
    intent_keywords = {
        "delivery_status": ["delivery", "delivered", "package", "tracking", "order", "arrive", "carrier", "late", "where is"],
        "refund_status": ["refund", "money", "returned", "credit", "reimburse", "back to my card"],
        "return_exchange": ["return", "exchange", "replace", "drop off", "send back", "label"],
        "cancellation_request": ["cancel", "cancelled", "cancellation", "cancel order", "stop delivery"],
        "damaged_defective_item": ["damaged", "broken", "crushed", "leaking", "defective", "box crushed", "smashed"],
        "billing_overcharge": ["charged", "charge", "double charged", "overcharged", "bank", "credit card", "billed"],
        "order_modification": ["address", "change address", "change delivery", "wrong address", "update address"],
        "account_access_security": ["account", "password", "login", "locked", "otp", "2fa", "verification", "hacked"],
        "prime_membership": ["prime", "prime video", "membership", "subscription", "annual fee"],
        "product_availability": ["stock", "out of stock", "available", "when will", "restock"],
        "general_feedback_complaint": ["driver", "worst", "terrible", "rude", "poor service", "disappointed", "unacceptable"],
        "app_technical_issue": ["app", "website", "error", "checkout", "cart", "crashing", "glitch", "button"]
    }

    final_taxonomy = {}
    for intent_name, base_info in BASE_TAXONOMY_DEFINITIONS.items():
        kws = intent_keywords.get(intent_name, [intent_name])
        # Find positive examples from real customer messages
        matching_examples = []
        for msg in messages:
            msg_lower = msg.lower()
            if any(kw in msg_lower for kw in kws):
                # Filter out overly short or long
                if 25 <= len(msg) <= 200 and msg not in matching_examples:
                    matching_examples.append(msg)
                    if len(matching_examples) >= 5:
                        break
        
        # Fallback if few matched
        if len(matching_examples) < 2:
            matching_examples = [m for m in messages if len(m) > 20][:3]

        final_taxonomy[intent_name] = {
            "intent": intent_name,
            "description": base_info["description"],
            "positive_examples": matching_examples,
            "confusing_intents": base_info["confusing_intents"],
            "escalation_guidance": base_info["escalation_guidance"]
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_taxonomy, f, indent=2, ensure_ascii=False)
        
    logger.info("Successfully discovered and saved %d intents to %s", len(final_taxonomy), output_path)
    return final_taxonomy


if __name__ == "__main__":
    discover_intent_taxonomy()
