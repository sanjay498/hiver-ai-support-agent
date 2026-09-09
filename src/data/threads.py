"""
Conversation Thread Reconstruction Module.
Reconstructs multi-turn Twitter customer support conversations:
1. Links tweets via in_response_to_tweet_id and response_tweet_id.
2. Identifies customer initial inquiries and subsequent brand agent replies.
3. Preserves chronological order and preceding conversation context.
4. Outputs structured conversation units for retrieval and evaluation.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("src.data.threads")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def normalize_text(text: str) -> str:
    """
    Cleans up Twitter handles and excess whitespace while preserving
    order IDs, punctuation, sentiment, and core conversational tokens.
    """
    if not isinstance(text, str):
        return ""
    # Strip leading @handles that Twitter adds automatically to replies
    cleaned = re.sub(r"^(@\w+\s*)+", "", text)
    # Replace multiple whitespaces/newlines with single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def reconstruct_conversations(
    df: pd.DataFrame,
    selected_brand: str = "AmazonHelp",
    max_conversations: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Reconstructs conversation pairs/threads for the given brand.
    Returns a list of conversation dictionaries.
    """
    logger.info("Reconstructing conversations for brand '%s'...", selected_brand)

    # Coerce columns
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df = df.dropna(subset=["tweet_id", "text"])
    df["tweet_id"] = df["tweet_id"].astype(int)
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
    df["author_id"] = df["author_id"].astype(str)
    
    # Clean text
    df["clean_text"] = df["text"].apply(normalize_text)
    
    # Fast lookup dictionaries
    tweet_records = df.set_index("tweet_id").to_dict(orient="index")
    
    # Find all replies from the brand
    brand_replies = df[(df["author_id"].str.lower() == selected_brand.lower()) & (~df["inbound"])]
    logger.info("Found %d support replies from brand '%s'", len(brand_replies), selected_brand)
    
    conversations = []
    seen_customer_texts = set()

    for _, brand_row in brand_replies.iterrows():
        brand_tweet_id = int(brand_row["tweet_id"])
        parent_id = brand_row.get("in_response_to_tweet_id")
        
        if pd.isna(parent_id):
            continue
        try:
            parent_id = int(float(parent_id))
        except (ValueError, TypeError):
            continue
            
        if parent_id not in tweet_records:
            continue
            
        parent_record = tweet_records[parent_id]
        # Check if parent is inbound (customer)
        if not parent_record.get("inbound", False):
            continue
            
        customer_text = parent_record.get("clean_text", "").strip()
        brand_reply = brand_row.get("clean_text", "").strip()
        
        if not customer_text or not brand_reply:
            continue
            
        # Deduplication check: ignore duplicate identical customer queries
        cust_norm_key = customer_text.lower()
        if cust_norm_key in seen_customer_texts:
            continue
        seen_customer_texts.add(cust_norm_key)
        
        # Build context if parent also replied to an earlier tweet
        context_history = []
        grandparent_id = parent_record.get("in_response_to_tweet_id")
        if pd.notna(grandparent_id):
            try:
                gp_id = int(float(grandparent_id))
                if gp_id in tweet_records:
                    gp_rec = tweet_records[gp_id]
                    context_history.append({
                        "tweet_id": gp_id,
                        "author_id": gp_rec["author_id"],
                        "text": gp_rec["clean_text"],
                        "inbound": gp_rec["inbound"]
                    })
            except (ValueError, TypeError):
                pass
                
        conversation_id = f"conv_{parent_id}_{brand_tweet_id}"
        conversations.append({
            "conversation_id": conversation_id,
            "customer_tweet_id": parent_id,
            "customer_author_id": str(parent_record.get("author_id", "customer")),
            "customer_text": customer_text,
            "brand_tweet_id": brand_tweet_id,
            "brand_author_id": selected_brand,
            "brand_reply": brand_reply,
            "context_history": context_history,
            "thread_length": 2 + len(context_history),
            "created_at": str(parent_record.get("created_at", ""))
        })
        
        if max_conversations and len(conversations) >= max_conversations:
            break

    logger.info("Successfully reconstructed %d valid conversation pairs for '%s'", len(conversations), selected_brand)
    return conversations


def save_conversations(conversations: List[Dict[str, Any]], output_json: Path, output_csv: Optional[Path] = None):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d conversations to %s", len(conversations), output_json)
    
    if output_csv:
        df_out = pd.DataFrame(conversations)
        # Convert list of dicts to string for CSV compatibility
        df_out["context_history"] = df_out["context_history"].apply(json.dumps)
        df_out.to_csv(output_csv, index=False)
        logger.info("Saved CSV copy to %s", output_csv)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct conversation threads.")
    parser.add_argument("--input-csv", type=str, default="data/raw/twcs_sample.csv")
    parser.add_argument("--brand", type=str, default=os.getenv("SELECTED_BRAND", "AmazonHelp"))
    parser.add_argument("--output-json", type=str, default="data/processed/conversations.json")
    parser.add_argument("--output-csv", type=str, default="data/processed/conversations.csv")
    parser.add_argument("--max-conversations", type=int, default=None)
    args = parser.parse_args()
    
    df_raw = pd.read_csv(args.input_csv, low_memory=False, on_bad_lines="skip")
    convs = reconstruct_conversations(df_raw, selected_brand=args.brand, max_conversations=args.max_conversations)
    save_conversations(convs, Path(args.output_json), Path(args.output_csv))
