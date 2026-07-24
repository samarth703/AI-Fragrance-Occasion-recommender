"""
03_train_model.py
------------------
AI-Powered Fragrance Occasion Recommender
Step 3: Train & Evaluate ML Model

Trains a multi-class classifier that predicts the best-suited event
(Office, Wedding, Date Night, Party, etc.) for a fragrance based on
its category, longevity, sillage, time of day, season, and price tier.
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/occasion_clean.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

feature_cols = ["brand", "category", "longevity", "sillage",
                 "best_time_of_day", "best_season", "price_tier"]
target_col = "recommended_event"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)]
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
best_model_name = None
best_model_pipeline = None
best_acc = 0

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # cross-validation for a more robust estimate given small dataset
    cv_scores = cross_val_score(pipe, X, y, cv=5)

    results[name] = acc
    print(f"\n--- {name} ---")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, preds, zero_division=0))

    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model_pipeline = pipe
        best_preds = preds

print(f"\n{'='*50}")
print(f"Best model: {best_model_name} (test accuracy={best_acc:.4f})")
print(f"{'='*50}")

joblib.dump(best_model_pipeline, os.path.join(MODEL_DIR, "occasion_classifier.pkl"))
df.to_csv(os.path.join(MODEL_DIR, "catalog.csv"), index=False)
print(f"Saved classifier -> {MODEL_DIR}/occasion_classifier.pkl")

# Confusion matrix plot for the best model
plt.figure(figsize=(9, 7))
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_preds, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual Event")
plt.xlabel("Predicted Event")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
os.makedirs("static", exist_ok=True)
plt.savefig("static/confusion_matrix.png", dpi=120)
plt.close()
print("Saved confusion matrix -> static/confusion_matrix.png")

# Model comparison bar chart
plt.figure(figsize=(7, 5))
plt.bar(results.keys(), results.values(), color=["#8e44ad", "#2980b9", "#c0392b"])
plt.title("Model Comparison - Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, (k, v) in enumerate(results.items()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig("static/model_comparison.png", dpi=120)
plt.close()
print("Saved model comparison chart -> static/model_comparison.png")

print("\nModel training complete.")
