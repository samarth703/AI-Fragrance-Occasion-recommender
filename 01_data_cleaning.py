"""
01_data_cleaning.py
--------------------
AI-Powered Fragrance Occasion Recommender
Step 1: Data Cleaning & Preprocessing
"""

import pandas as pd
import os

RAW_PATH = "data/perfume_occasion_dataset.csv"
CLEAN_PATH = "data/occasion_clean.csv"

def load_data(path):
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")
    return df

def check_missing(df):
    nulls = df.isnull().sum()
    print("\nMissing values per column:")
    print(nulls)
    return df

def standardize_text(df):
    text_cols = ["brand", "category", "longevity", "sillage",
                 "best_time_of_day", "best_season", "price_tier", "recommended_event"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df

def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nDropped {before - len(df)} duplicate rows")
    return df

def encode_ordinal_features(df):
    """Longevity and sillage have a natural order - encode ordinally
    in addition to keeping the original text label (useful for EDA)."""
    longevity_order = {"Light": 1, "Medium": 2, "Strong": 3, "Very Strong": 4}
    sillage_order = {"Intimate": 1, "Moderate": 2, "Strong": 3, "Enormous": 4}

    df["longevity_score"] = df["longevity"].map(longevity_order)
    df["sillage_score"] = df["sillage"].map(sillage_order)
    return df

def validate_categories(df):
    """Sanity check: flag any unexpected category values."""
    expected_events = {"Office","Casual Daywear","Date Night","Wedding","Party/Club",
                        "Formal Event","Gym/Sports","Beach/Vacation","Night Out"}
    unexpected = set(df["recommended_event"].unique()) - expected_events
    if unexpected:
        print(f"\nWarning: unexpected event labels found: {unexpected}")
    else:
        print("\nAll recommended_event labels are within expected set.")
    return df

def main():
    df = load_data(RAW_PATH)
    df = check_missing(df)
    df = standardize_text(df)
    df = drop_duplicates(df)
    df = encode_ordinal_features(df)
    df = validate_categories(df)

    print(f"\nFinal cleaned shape: {df.shape}")
    print("\nEvent distribution:")
    print(df["recommended_event"].value_counts())

    os.makedirs("data", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_PATH}")

if __name__ == "__main__":
    main()
