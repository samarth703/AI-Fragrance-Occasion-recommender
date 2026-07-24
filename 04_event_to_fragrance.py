"""
04_event_to_fragrance.py
--------------------------
AI-Powered Fragrance Occasion Recommender
Reverse Direction: Event -> Fragrance Detection

Given an event (Wedding, Office, Date Night, etc.) plus optional
season/time-of-day/price preference, this predicts the best-matching
fragrance CATEGORY (scent family) and returns specific recommended
fragrances from the catalog that match.

This complements 03_train_model.py (which goes fragrance -> event) by
solving the inverse problem: event -> fragrance.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/occasion_clean.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------
# Reverse Classifier: event (+context) -> fragrance category
# ---------------------------------------------------------------
feature_cols = ["recommended_event", "best_season", "best_time_of_day", "price_tier"]
target_col = "category"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)]
)

clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
acc = accuracy_score(y_test, preds)
cv_scores = cross_val_score(pipe, X, y, cv=5)

print("=== Event -> Fragrance Category Classifier ===")
print(f"Test Accuracy: {acc:.4f}")
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(classification_report(y_test, preds, zero_division=0))

joblib.dump(pipe, os.path.join(MODEL_DIR, "event_to_category_classifier.pkl"))
print(f"Saved -> {MODEL_DIR}/event_to_category_classifier.pkl")

# Confusion matrix
plt.figure(figsize=(11, 8))
labels = sorted(y.unique())
cm = confusion_matrix(y_test, preds, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Event to Fragrance Category")
plt.ylabel("Actual Category")
plt.xlabel("Predicted Category")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
os.makedirs("static", exist_ok=True)
plt.savefig("static/event_to_category_confusion_matrix.png", dpi=120)
plt.close()
print("Saved confusion matrix -> static/event_to_category_confusion_matrix.png")


def detect_fragrances_for_event(event, season=None, time_of_day=None, price_tier=None, top_n=5):
    """
    Given an event (+ optional context), predicts the best fragrance
    category, then returns top_n specific fragrances from the catalog
    that match that event (and context filters, if provided).
    """
    input_row = pd.DataFrame([{
        "recommended_event": event,
        "best_season": season or df["best_season"].mode()[0],
        "best_time_of_day": time_of_day or df["best_time_of_day"].mode()[0],
        "price_tier": price_tier or df["price_tier"].mode()[0],
    }])

    predicted_category = pipe.predict(input_row)[0]
    probs = pipe.predict_proba(input_row)[0]
    top_categories = dict(sorted(zip(pipe.classes_, probs), key=lambda x: x[1], reverse=True)[:3])

    # Filter catalog for actual matching fragrances
    matches = df[df["recommended_event"] == event].copy()
    if season:
        season_matches = matches[matches["best_season"] == season]
        if not season_matches.empty:
            matches = season_matches
    if price_tier:
        price_matches = matches[matches["price_tier"] == price_tier]
        if not price_matches.empty:
            matches = price_matches

    matches = matches.sort_values("sillage_score", ascending=False).head(top_n)

    return {
        "predicted_top_category": predicted_category,
        "category_confidence": {k: round(v * 100, 1) for k, v in top_categories.items()},
        "recommended_fragrances": matches[["brand", "category", "longevity", "sillage", "price_tier"]].to_dict(orient="records")
    }


print("\n--- Sample Test: Wedding ---")
result = detect_fragrances_for_event("Wedding", season="Fall", price_tier="Luxury")
print("Predicted category:", result["predicted_top_category"])
print("Confidence:", result["category_confidence"])
print("Recommended fragrances:")
for r in result["recommended_fragrances"]:
    print(" -", r)

print("\nEvent-to-Fragrance detection model training complete.")
