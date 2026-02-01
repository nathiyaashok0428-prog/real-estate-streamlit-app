# ============================================================
# Real Estate Investment Advisor - Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    clf = joblib.load("models/classification_model.pkl")
    reg = joblib.load("models/regression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return clf, reg, scaler

clf_model, reg_model, scaler = load_models()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_real_estate_data_small.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Introduction", "EDA Visualizations", "Prediction"]
)

# ============================================================
# INTRODUCTION PAGE
# ============================================================
if page == "Introduction":
    st.title("🏘️ Real Estate Investment Advisor")
    st.subheader("Predicting Property Profitability & Future Value")

    st.markdown("""
    **Project Objective**
    - Identify whether a property is a *Good Investment*
    - Predict estimated future price
    - Provide data-driven insights for investors

    **Technologies Used**
    - Python, Pandas, Scikit-learn
    - MLflow
    - Streamlit
    """)

# ============================================================
# EDA PAGE
# ============================================================
elif page == "EDA Visualizations":
    st.title("📊 Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Property Prices")
        fig, ax = plt.subplots()
        sns.histplot(df["Price_in_Lakhs"], bins=40, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Distribution of Property Size")
        fig, ax = plt.subplots()
        sns.histplot(df["Size_in_SqFt"], bins=40, kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Price vs Size")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["Size_in_SqFt"],
        y=df["Price_in_Lakhs"],
        alpha=0.3
    )
    st.pyplot(fig)

# ============================================================
# PREDICTION PAGE
# ============================================================
elif page == "Prediction":
    st.title("📈 Property Investment Prediction")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            city = st.selectbox("City", sorted(df["City"].unique()))
            bhk = st.number_input("BHK", 1, 10, 2)
            size = st.number_input("Size (Sq Ft)", 200, 10000, 1000)

        with col2:
            price = st.number_input("Current Price (Lakhs)", 5.0, 500.0, 50.0)
            age = st.number_input("Age of Property (Years)", 0, 50, 5)
            floor = st.number_input("Floor No", 0, 50, 2)

        with col3:
            schools = st.number_input("Nearby Schools", 0, 20, 2)
            hospitals = st.number_input("Nearby Hospitals", 0, 20, 1)
            total_floors = st.number_input("Total Floors", 1, 60, 10)

        submit = st.form_submit_button("Predict")

    if submit:
        price_per_sqft = (price * 100000) / size

        input_data = pd.DataFrame([[
            bhk, size, price_per_sqft, age,
            schools, hospitals, floor, total_floors
        ]], columns=[
            "BHK", "Size_in_SqFt", "Price_per_SqFt",
            "Age_of_Property", "Nearby_Schools",
            "Nearby_Hospitals", "Floor_No", "Total_Floors"
        ])

        input_scaled = scaler.transform(input_data)

        # Classification
        invest_pred = clf_model.predict(input_scaled)[0]
        invest_prob = clf_model.predict_proba(input_scaled)[0][1]

        # Regression
        future_price = reg_model.predict(input_scaled)[0]

        # ---------------- RESULTS ----------------
        st.subheader("📌 Investment Result")

        if invest_pred == 1:
            st.success("✅ Good Investment")
        else:
            st.error("❌ Not a Good Investment")

        st.info(f"Model Confidence: {round(invest_prob*100, 2)} %")

        st.subheader("💰 Estimated Future Price (5 Years)")
        st.metric(
            label="Predicted Price (Lakhs)",
            value=f"₹ {round(future_price, 2)}"
        )

        growth = ((future_price - price) / price) * 100
        st.metric(
            label="Expected Growth (%)",
            value=f"{round(growth, 2)} %"
        )
