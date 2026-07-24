"""
02_eda.py
---------
AI-Powered Fragrance Occasion Recommender
Step 2: Exploratory Data Analysis
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
os.makedirs("static", exist_ok=True)

df = pd.read_csv("data/occasion_clean.csv")

print("=== Basic Info ===")
print(df.info())
print(f"\nTotal records: {len(df)}")
print(f"Unique brands: {df['brand'].nunique()}")
print(f"Unique categories: {df['category'].nunique()}")
print(f"Unique events: {df['recommended_event'].nunique()}")

# 1. Event distribution
plt.figure(figsize=(9, 5))
order = df["recommended_event"].value_counts().index
sns.countplot(data=df, y="recommended_event", order=order,
               hue="recommended_event", palette="magma", legend=False)
plt.title("Distribution of Recommended Events")
plt.tight_layout()
plt.savefig("static/event_distribution.png", dpi=120)
plt.close()

# 2. Category vs Event heatmap
pivot = pd.crosstab(df["category"], df["recommended_event"])
plt.figure(figsize=(11, 7))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Fragrance Category vs Recommended Event")
plt.tight_layout()
plt.savefig("static/category_vs_event_heatmap.png", dpi=120)
plt.close()

# 3. Longevity/Sillage score by event (boxplots)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="recommended_event", y="longevity_score", ax=axes[0],
            hue="recommended_event", palette="crest", legend=False)
axes[0].set_title("Longevity Score by Event")
axes[0].tick_params(axis="x", rotation=45)

sns.boxplot(data=df, x="recommended_event", y="sillage_score", ax=axes[1],
            hue="recommended_event", palette="flare", legend=False)
axes[1].set_title("Sillage Score by Event")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("static/longevity_sillage_by_event.png", dpi=120)
plt.close()

# 4. Season vs Event
plt.figure(figsize=(10, 6))
pivot2 = pd.crosstab(df["best_season"], df["recommended_event"])
sns.heatmap(pivot2, annot=True, fmt="d", cmap="BuPu")
plt.title("Best Season vs Recommended Event")
plt.tight_layout()
plt.savefig("static/season_vs_event_heatmap.png", dpi=120)
plt.close()

# 5. Price tier distribution across events
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="recommended_event", hue="price_tier", palette="Set2")
plt.title("Price Tier Distribution across Events")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/price_tier_by_event.png", dpi=120)
plt.close()

print("\nAll EDA plots saved to static/")
