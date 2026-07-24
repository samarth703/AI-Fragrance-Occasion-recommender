# 🎭 AI-Powered Fragrance Occasion Recommender (Streamlit Edition)

A solo ML capstone project with **two-way fragrance ↔ event intelligence**,
deployed as a Streamlit web app.

1. **Fragrance → Event** — given a fragrance's attributes (category,
   longevity, sillage, etc.), predicts the best-suited event to wear it to.
2. **Event → Fragrance (Detection)** — given an event, detects the
   best-fit fragrance family and recommends specific matching fragrances.

## Dataset
`data/perfume_occasion_dataset.csv` — 320 fragrance records with category,
performance attributes (longevity/sillage), and a target `recommended_event`
label.

> **Data provenance note:** synthetically generated with rule-based
> category→event logic plus ~7% realistic label noise, not scraped from a
> real source. Mention this transparently in your project report.

## Project Structure
```
streamlit_fragrance_project/
├── data/
│   ├── perfume_occasion_dataset.csv   # raw data
│   └── occasion_clean.csv             # cleaned data (generated)
├── models/                            # trained models + catalog (generated)
├── static/                            # EDA + evaluation plots (generated)
├── 01_data_cleaning.py                # Step 1: cleaning & preprocessing
├── 02_eda.py                          # Step 2: exploratory data analysis
├── 03_train_model.py                  # Step 3a: fragrance -> event model
├── 04_event_to_fragrance.py           # Step 3b: event -> fragrance model
├── streamlit_app.py                   # Step 4: Streamlit deployment
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt

python 01_data_cleaning.py         # Step 1: clean the data
python 02_eda.py                   # Step 2: generate EDA plots -> static/
python 03_train_model.py           # Step 3a: train fragrance->event model
python 04_event_to_fragrance.py    # Step 3b: train event->fragrance model

streamlit run streamlit_app.py     # Step 4: launch the app
```
The app opens automatically at `http://localhost:8501`.

## What the App Does

The app has **two tabs**:

**Tab 1 — 🧴 Fragrance → Event**
Select brand, category, longevity, sillage, time of day, season, price tier
→ get the predicted best occasion with a full confidence breakdown across
all 8 event classes.

**Tab 2 — 🎉 Event → Fragrance**
Select an event (+ optional season/price tier) → the app detects the
best-fit fragrance category and lists specific recommended fragrances
from the catalog.

## Model Results

**1. Fragrance → Event Classifier**
- Multi-class classification, 8 event classes
- Models compared: Random Forest, Logistic Regression, SVM (5-fold CV)
- **Best: Random Forest — 92.2% test accuracy, 94.4% CV accuracy**

**2. Event → Fragrance Detector (reverse)**
- Multi-class classification predicting fragrance category from event context
- **Random Forest — 67.2% test accuracy, 70.6% CV accuracy**
  (lower than the forward model is expected: some events like "Wedding"
  genuinely map to more than one fragrance family — real ambiguity,
  not a modeling flaw)

## Data Cleaning Steps
- Verified no missing values
- Standardized text casing across categorical columns
- Removed exact duplicates
- Added ordinal `longevity_score` / `sillage_score` encodings
- Validated all event labels against the expected set

## Notes for Submission
- Solo project, unique topic — a two-way fragrance↔occasion system is
  uncommon among student projects.
- Be transparent that the dataset is synthetically generated with
  realistic rule-based structure, not scraped.
