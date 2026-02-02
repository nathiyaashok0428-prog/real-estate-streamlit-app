🏡 Real Estate Investment Advisor
Predicting Property Profitability & Future Value

🔗 Live App (Streamlit Cloud):
👉 (Paste your Streamlit URL here)

📌 Project Overview

The Real Estate Investment Advisor is an end-to-end Data Science & Machine Learning project that helps users:

✅ Classify whether a property is a Good Investment

📈 Predict future property price (after 5 years)

📊 Explore interactive EDA visualizations

🧠 Support data-driven real estate decisions

This project covers the complete ML lifecycle:
Data Cleaning → EDA → Feature Engineering → ML Models → Deployment.

🎯 Problem Statement

Real estate investors often struggle to:

Identify profitable properties

Understand pricing patterns across cities

Estimate future property value

This project solves that by combining historical housing data with machine learning models and an interactive Streamlit app.

🧱 Project Architecture
real_estate_streamlit_FINAL/
│
├── app.py                      # Streamlit application
├── requirements.txt            # Dependencies for Streamlit Cloud
├── README.md                   # Project documentation
│
├── data/
│   └── cleaned_real_estate_data_small.csv
│
├── models/
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl

🛠️ Tech Stack & Skills Used
Programming & Data

Python

Pandas, NumPy

Data Cleaning & Feature Engineering

Exploratory Data Analysis (EDA)

Machine Learning

Logistic Regression

Random Forest Classifier

Random Forest Regressor

Feature Scaling (StandardScaler)

Model Evaluation (Accuracy, R², MAE)

Visualization

Matplotlib

Seaborn

Streamlit Charts

Deployment

Streamlit Cloud

Git & GitHub

📊 Exploratory Data Analysis (EDA)

The app includes multiple EDA insights, such as:

📍 Average Price per SqFt by City

📐 Distribution of Property Sizes

🏢 BHK vs Property Price

⏳ Property Age vs Price

🏙️ City-wise Price Comparison

📈 Correlation between features

All EDA charts are interactive and rendered live in the app.

🤖 Machine Learning Models
🔹 Classification (Good Investment?)

Target: good_investment (0 / 1)

Model: Random Forest Classifier

Output:

Good / Not Good Investment

Model confidence (%)

🔹 Regression (Future Price Prediction)

Target: future_price_5yrs

Model: Random Forest Regressor

Output:

Estimated price after 5 years (₹ Lakhs)

✔ Lightweight models used for cloud deployment compatibility

🖥️ Streamlit App Features
🔹 Pages

Introduction – Project overview

EDA Visualizations – Data insights

Prediction – ML-based investment decision

🔹 User Inputs

City

BHK

Size (Sq Ft)

Property Age

Floor No & Total Floors

Nearby Schools & Hospitals

🔹 Outputs

✅ Investment Decision

📊 Model Confidence

💰 Estimated Future Price (5 Years)

🚀 Deployment

The app is deployed on Streamlit Cloud.

Deployment Highlights

Uses small cleaned dataset

Lightweight .pkl models

Optimized for GitHub file limits

Fully cloud-ready setup

⚠️ Limitations

Dataset is synthetic (for learning purposes)

Market fluctuations not modeled

External economic factors not included

🔮 Future Improvements

Add real-time market data

Use time-series forecasting

Integrate map-based visualizations

Improve model calibration

Add user authentication

Handling large files for cloud deployment

Streamlit deployment challenges & fixes

Model confidence interpretation

👩‍💻 Author

Nathiya Ashok
📧 nathiyaashok0428@gmail.com
🔗 GitHub: [https://github.com/nathiyaashok0428-prog](https://github.com/nathiyaashok0428-prog/real-estate-streamlit-app)
