import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# Load and train model
@st.cache_resource
def load_model():
    # Check if file exists
    data_path = "heart_cleveland_upload.csv"
    if not os.path.exists(data_path):
        st.error(f"Dataset file '{data_path}' not found. Please upload it to the project directory.")
        return None

    # Load data
    df = pd.read_csv(data_path)
    
    # Replace '?' with NaN and drop rows with missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Features and target
    X = df.drop(columns=['condition'])
    y = df['condition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    return model

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("Enter patient information to predict the risk of heart disease.")

if model is None:
    st.stop()

inputs = {}
fields = [
    ("Age", "age"),
    ("Sex (1 = male, 0 = female)", "sex"),
    ("Chest Pain Type (0–3)", "cp"),
    ("Resting Blood Pressure", "trestbps"),
    ("Cholesterol", "chol"),
    ("Fasting Blood Sugar > 120 (1 = yes, 0 = no)", "fbs"),
    ("Resting ECG (0–2)", "restecg"),
    ("Max Heart Rate Achieved", "thalach"),
    ("Exercise-Induced Angina (1 = yes, 0 = no)", "exang"),
    ("Oldpeak (ST depression)", "oldpeak"),
    ("Slope (0–2)", "slope"),
    ("Number of Major Vessels (0–3)", "ca"),
    ("Thal (1 = normal, 2 = fixed defect, 3 = reversible defect)", "thal")
]

for label, key in fields:
    inputs[key] = st.number_input(label, value=0.0, step=0.1)

if st.button("Predict"):
    input_vector = np.array([inputs[key] for _, key in fields]).reshape(1, -1)
    prediction = model.predict(input_vector)[0]
    if prediction == 1.0:
        st.error("⚠️ The model predicts: The patient **has** heart disease.")
    else:
        st.success("✅ The model predicts: The patient **does not** have heart disease.")

# Footer
st.markdown(
    "<hr style='margin-top: 50px;'>"
    "<div style='text-align: center;'>"
    "👨‍💻 Developed by <a href='https://www.linkedin.com/in/youssef-abdalnasser-33705b262/' target='_blank'>Youssef Abdelnasser</a>"
    "</div>",
    unsafe_allow_html=True
)
