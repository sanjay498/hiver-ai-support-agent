"""
Data Cleaning & Preprocessing Pipeline.
Coordinates:
1. Loading raw dataset sample.
2. Filtering and normalizing tweets for selected brand.
3. Thread reconstruction into customer-brand conversational units.
4. Outputting cleaned data artifacts for intent discovery and retrieval indexing.
"""

import os
import argparse
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from src.data.threads import reconstruct_conversations, save_conversations

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.data.clean")


def clean_and_process(
    input_csv: Path = Path("data/raw/twcs_sample.csv"),
    brand: str = "AmazonHelp",
    output_dir: Path = Path("data/processed"),
    max_conversations: int = 5000,
    random_seed: int = 42
):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_csv}. Run 'python -m src.data.download' first.")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading raw dataset from %s ...", input_csv)
    df_raw = pd.read_csv(input_csv, low_memory=False, on_bad_lines="skip")
    logger.info("Loaded raw dataset: %d rows, %d columns", len(df_raw), len(df_raw.columns))
    
    # Reconstruct conversations
    conversations = reconstruct_conversations(
        df=df_raw,
        selected_brand=brand,
        max_conversations=max_conversations
    )
    
    if not conversations:
        raise ValueError(f"No conversations found for brand '{brand}'. Please verify the raw dataset contains tweets for this brand.")
        
    output_json = output_dir / "conversations.json"
    output_csv = output_dir / "conversations.csv"
    save_conversations(conversations, output_json, output_csv)
    
    # Summary stats
    lengths = [c["thread_length"] for c in conversations]
    cust_chars = [len(c["customer_text"]) for c in conversations]
    brand_chars = [len(c["brand_reply"]) for c in conversations]
    
    logger.info("=== Preprocessing Summary for '%s' ===", brand)
    logger.info("Total Reconstructed Conversations: %d", len(conversations))
    logger.info("Avg Thread Length: %.2f turns", sum(lengths) / len(lengths))
    logger.info("Avg Customer Message Length: %.1f characters", sum(cust_chars) / len(cust_chars))
    logger.info("Avg Brand Reply Length: %.1f characters", sum(brand_chars) / len(brand_chars))
    logger.info("Processed artifacts successfully generated in %s", output_dir)
    return conversations


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and reconstruct customer support conversations.")
    parser.add_argument("--input-csv", type=str, default="data/raw/twcs_sample.csv",
                        help="Path to raw twcs_sample.csv.")
    parser.add_argument("--brand", type=str, default=os.getenv("SELECTED_BRAND", "AmazonHelp"),
                        help="Brand name to filter (default: AmazonHelp).")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for processed files.")
    parser.add_argument("--max-conversations", type=int, default=int(os.getenv("MAX_CONVERSATIONS", "5000")),
                        help="Maximum conversations to extract.")
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")),
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    clean_and_process(
        input_csv=Path(args.input_csv),
        brand=args.brand,
        output_dir=Path(args.output_dir),
        max_conversations=args.max_conversations,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
