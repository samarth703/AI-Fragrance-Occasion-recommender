import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Fragrance Occasion Recommender",
    page_icon="🧴",
    layout="wide"
)

# -----------------------------
# Deep Emerald & Burgundy Luxury Theme
# -----------------------------
st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#0B3D2E,#1F2937,#4C1D24);
}

/* Main Title */
h1{
    color:#F4D9B0;
    text-align:center;
    font-size:42px;
    font-weight:bold;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
}

/* Sub Heading */
h2,h3{
    color:#E8B4B8;
}

/* Normal Text */
p,label,span,div{
    color:#F5EFE6;
    font-size:16px;
}

/* Dropdown Labels */
[data-testid="stWidgetLabel"] p{
    color:#F4D9B0 !important;
    font-weight:bold;
    font-size:16px;
}

/* Select Boxes */
.stSelectbox > div > div{
    background-color:#1B4332;
    border:1px solid #C9A66B;
    border-radius:10px;
    color:#F5EFE6;
}

/* Button */
div.stButton > button{
    background: linear-gradient(90deg,#7B1E3A,#9C2C4C);
    color:#F4D9B0;
    font-size:18px;
    font-weight:bold;
    border:1px solid #C9A66B;
    border-radius:12px;
    padding:12px;
    transition:0.3s;
}

div.stButton > button:hover{
    background: linear-gradient(90deg,#5C1630,#7B1E3A);
    color:#FFFFFF;
    transform:scale(1.03);
}

/* Success Box */
[data-testid="stAlert"]{
    background:#153226;
    border-left:6px solid #C9A66B;
    color:#F4D9B0;
}
[data-testid="stAlert"] p{
    color:#F4D9B0 !important;
}

/* DataFrame */
[data-testid="stDataFrame"]{
    border:2px solid #C9A66B;
    border-radius:12px;
    overflow:hidden;
}

/* Horizontal Line */
hr{
    border:1px solid #C9A66B;
}

/* Footer */
footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_DIR = "models"

try:
    catalog = pd.read_csv(os.path.join(MODEL_DIR, "catalog.csv"))
except FileNotFoundError:
    st.error("❌ models/catalog.csv not found.")
    st.stop()

try:
    classifier = joblib.load(os.path.join(MODEL_DIR, "occasion_classifier.pkl"))
except FileNotFoundError:
    st.error("❌ models/occasion_classifier.pkl not found.")
    st.stop()

# -----------------------------
# Dropdown Lists
# -----------------------------
BRAND_LIST = sorted(catalog["brand"].unique())
CATEGORY_LIST = sorted(catalog["category"].unique())

LONGEVITY_LIST = ["Light", "Medium", "Strong", "Very Strong"]
SILLAGE_LIST = ["Intimate", "Moderate", "Strong", "Enormous"]

TIME_LIST = sorted(catalog["best_time_of_day"].unique())
SEASON_LIST = sorted(catalog["best_season"].unique())

PRICE_TIER_LIST = ["Budget", "Mid-Range", "Luxury"]

# -----------------------------
# Title
# -----------------------------
st.title("⚡ AI-Powered Fragrance Occasion Recommender")

st.write(
    "Select the fragrance details below and click **Predict Occasion**."
)

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", BRAND_LIST)
    category = st.selectbox("Category", CATEGORY_LIST)
    longevity = st.selectbox("Longevity", LONGEVITY_LIST)
    sillage = st.selectbox("Sillage", SILLAGE_LIST)

with col2:
    time_of_day = st.selectbox("Best Time of Day", TIME_LIST)
    season = st.selectbox("Best Season", SEASON_LIST)
    price_tier = st.selectbox("Price Tier", PRICE_TIER_LIST)

# -----------------------------
# Prediction
# -----------------------------
if st.button("✨ Predict Occasion"):

    input_df = pd.DataFrame([{
        "brand": brand,
        "category": category,
        "longevity": longevity,
        "sillage": sillage,
        "best_time_of_day": time_of_day,
        "best_season": season,
        "price_tier": price_tier
    }])

    prediction = classifier.predict(input_df)[0]
    probs = classifier.predict_proba(input_df)[0]
    classes = classifier.classes_

    st.success(f"🎉 Recommended Occasion: **{prediction}**")

    probability_df = pd.DataFrame({
        "Occasion": classes,
        "Probability (%)": [round(p * 100, 2) for p in probs]
    })

    probability_df = probability_df.sort_values(
        by="Probability (%)",
        ascending=False
    )

    st.subheader("📊 Prediction Probability")

    st.dataframe(
        probability_df,
        use_container_width=True
    )

    st.bar_chart(
        probability_df.set_index("Occasion")
    )

st.markdown("---")
st.caption("🧴 AI-Powered Fragrance Occasion Recommender • Machine Learning Project")