import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide"
)

# ---------------- LOAD ASSETS ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_real_estate_data_small.csv")

@st.cache_resource
def load_models():
    reg_model = joblib.load("models/regression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return reg_model, scaler, feature_cols


df = load_data()
reg_model, scaler, feature_cols = load_models()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Introduction", "EDA Visualizations", "Prediction"]
)

# ---------------- INTRO ----------------
if page == "Introduction":
    st.title("🏠 Real Estate Investment Advisor")
    st.write("""
    This application helps investors:
    - Predict **future property price (5 years)**
    - Decide whether a property is a **Good Investment**
    - Explore **EDA insights**
    """)

# ---------------- EDA ----------------
elif page == "EDA Visualizations":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Average Price per SqFt by City")
    city_price = df.groupby("City")["Price_per_SqFt"].mean().sort_values()
    st.bar_chart(city_price)

# ---------------- PREDICTION ----------------
else:
    st.title("📈 Property Investment Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("City", sorted(df["City"].unique()))
        bhk = st.number_input("BHK", 1, 5, 2)
        size = st.number_input("Size (Sq Ft)", 300, 5000, 1000)

    with col2:
        age = st.number_input("Age of Property (Years)", 0, 50, 5)
        floor_no = st.number_input("Floor No", 0, 50, 2)
        total_floors = st.number_input("Total Floors", 1, 60, 10)

    with col3:
        schools = st.number_input("Nearby Schools", 0, 10, 2)
        hospitals = st.number_input("Nearby Hospitals", 0, 10, 1)

    if st.button("Predict"):

        # ---------------- BUILD INPUT ----------------
        input_data = {
            "BHK": bhk,
            "Size_SqFt": size,
            "Age_of_Property": age,
            "Floor_No": floor_no,
            "Total_Floors": total_floors,
            "Nearby_Schools": schools,
            "Nearby_Hospitals": hospitals,
            "Price_per_SqFt": df[df["City"] == city]["Price_per_SqFt"].mean(),
        }

        input_df = pd.DataFrame([input_data])

        # Align with training features
        for col in feature_cols:
            if col not in input_df:
                input_df[col] = 0

        input_df = input_df[feature_cols]

        # ---------------- PRICE PREDICTION ----------------
        input_scaled = scaler.transform(input_df)
        future_price = reg_model.predict(input_scaled)[0]

        # ---------------- RULE-BASED INVESTMENT LOGIC ----------------
        city_avg_price = df[df["City"] == city]["Price_per_SqFt"].mean()
        predicted_price_sqft = future_price / size

        good_investment = (
            predicted_price_sqft >= city_avg_price and
            age <= 10 and
            schools >= 3 and
            hospitals >= 2
        )

        # Confidence score (interpretable)
        confidence = min(
            0.95,
            (
                (predicted_price_sqft / city_avg_price) * 0.4 +
                (schools / 5) * 0.2 +
                (hospitals / 5) * 0.2 +
                ((10 - min(age, 10)) / 10) * 0.2
            )
        )

        # ---------------- OUTPUT ----------------
        if good_investment:
            st.success("✅ Good Investment")
        else:
            st.error("❌ Not a Good Investment")

        st.info(f"Model Confidence: {confidence * 100:.2f}%")

        st.subheader("Estimated Future Price (5 Years)")
        st.write(f"₹ {future_price:,.2f} Lakhs")
