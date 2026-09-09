"""Tests for Data Cleaning & Thread Reconstruction."""

import pytest
import pandas as pd
from src.data.threads import normalize_text, reconstruct_conversations


def test_normalize_text():
    raw = "@AmazonHelp @115821 My package is late!   Please help.  https://t.co/xyz"
    cleaned = normalize_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "My package is late! Please help. https://t.co/xyz" == cleaned


def test_reconstruct_conversations_mock():
    # Construct a minimal mock dataframe representing a Twitter conversation
    data = {
        "tweet_id": [101, 102],
        "author_id": ["cust_123", "AmazonHelp"],
        "inbound": [True, False],
        "created_at": ["Tue Oct 31 10:00:00 +0000 2017", "Tue Oct 31 10:05:00 +0000 2017"],
        "text": ["@AmazonHelp Where is my package?", "@cust_123 Please check tracking in Your Orders. ^CS"],
        "response_tweet_id": [102, None],
        "in_response_to_tweet_id": [None, 101]
    }
    df = pd.DataFrame(data)
    convs = reconstruct_conversations(df, selected_brand="AmazonHelp")
    assert len(convs) == 1
    c = convs[0]
    assert c["customer_tweet_id"] == 101
    assert c["brand_tweet_id"] == 102
    assert "Where is my package?" in c["customer_text"]
    assert "Please check tracking" in c["brand_reply"]
