import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= CONFIG =================
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# ================= LOAD MODELS =================
scaler = joblib.load("models/scaler.pkl")
clf_model = joblib.load("models/classification_model.pkl")
reg_model = joblib.load("models/regression_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_real_estate_data_small.csv")

df = load_data()

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Introduction", "EDA Visualizations", "Prediction"]
)

# ================= INTRO =================
if page == "Introduction":
    st.title("🏡 Real Estate Investment Advisor")
    st.write("""
    This application helps investors:
    - Classify **Good vs Bad Investments**
    - Predict **Future Property Price**
    - Explore **Market Trends via EDA**
    """)

# ================= EDA =================
elif page == "EDA Visualizations":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Price Distribution")
    st.bar_chart(df["Price_in_Lakhs"])

    st.subheader("Price per SqFt by City")
    city_price = df.groupby("City")["Price_per_SqFt"].mean()
    st.bar_chart(city_price)

# ================= PREDICTION =================
else:
    st.title("📈 Property Investment Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("City", df["City"].unique())
        bhk = st.number_input("BHK", 1, 10, 2)
        size = st.number_input("Size (Sq Ft)", 300, 5000, 1000)

    with col2:
        age = st.number_input("Age of Property (Years)", 0, 50, 5)
        floor_no = st.number_input("Floor No", 0, 50, 2)
        total_floors = st.number_input("Total Floors", 1, 50, 10)

    with col3:
        schools = st.number_input("Nearby Schools", 0, 10, 2)
        hospitals = st.number_input("Nearby Hospitals", 0, 10, 1)

    if st.button("Predict"):
        # ========== INPUT DATAFRAME ==========
        input_dict = {
            "City": city,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Age_of_Property": age,
            "Nearby_Schools": schools,
            "Nearby_Hospitals": hospitals,
            "Floor_No": floor_no,
            "Total_Floors": total_floors,
        }

        input_df = pd.DataFrame([input_dict])

        # ========== ENCODE ==========
        input_df = pd.get_dummies(input_df, drop_first=True)

        # ========== ALIGN FEATURES ==========
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # ========== SCALE ==========
        input_scaled = scaler.transform(input_df)

        # ========== PREDICT ==========
        invest_pred = clf_model.predict(input_scaled)[0]
        invest_prob = clf_model.predict_proba(input_scaled)[0][1]
        future_price = reg_model.predict(input_scaled)[0]

        # ========== OUTPUT ==========
        if invest_pred == 1:
            st.success("✅ Good Investment")
        else:
            st.error("❌ Not a Good Investment")

        st.info(f"Model Confidence: {invest_prob*100:.2f}%")
        st.metric("Estimated Future Price (5 Years)", f"₹ {future_price:.2f} Lakhs")
