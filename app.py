import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="❤️")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px; font-size: 18px; padding: 10px;}
    .stSlider>div>div>div {background-color: #4CAF50;}
    .stSelectbox>div>div>select {background-color: #ffffff; border-radius: 5px; padding: 5px;}
    .header {color: #2E7D32; font-size: 36px; text-align: center; font-weight: bold;}
    .subheader {color: #388E3C; font-size: 24px;}
    .footer {text-align: center; color: #757575; margin-top: 50px;}
    .prediction-box {padding: 20px; border-radius: 10px; margin: 20px 0;}
    </style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<p class="header">🔍 Heart Disease Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="footer">Created by <a href="https://www.linkedin.com/in/youssef-abdalnasser-33705b262/" target="_blank">Youssef Abdelnasser</a></p>',
    unsafe_allow_html=True
)

# Load model from Google Drive
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    link_file = "model_link.txt"
    
    if not os.path.exists(model_path):
        if os.path.exists(link_file):
            with open(link_file, 'r') as f:
                model_url = f.read().strip().split(': ')[1]
            try:
                gdown.download(model_url, model_path, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None
        else:
            st.error(f"Model link file '{link_file}' not found.")
            return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Input form
st.markdown('<p class="subheader">Enter Patient Data</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50, help="Patient's age in years")
    sex = st.selectbox("Sex", ["Male", "Female"], help="Patient's gender")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Resting blood pressure in mm Hg")
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240, help="Serum cholesterol level")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], help="1: Yes, 0: No")
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")

with col2:
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 220, 150, help="Maximum heart rate during exercise")
    exang = st.selectbox("Exercise Induced Angina", [0, 1], help="1: Yes, 0: No")
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3], help="Number of major vessels (0-3)")
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3], help="0: None, 1: Normal, 2: Fixed defect, 3: Reversible defect")

# Prediction and Visualizations
if st.button("Predict", key="predict"):
    if model is None:
        st.error("Model not loaded. Please check the model link or file.")
    else:
        try:
            input_data = [[
                age,
                1 if sex == "Male" else 0,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]]
            prediction = model.predict(input_data)
            
            # Display prediction
            if prediction[0] == 1:
                st.markdown('<div class="prediction-box" style="background-color: #FFCDD2;"><h3>⚠️ Prediction: <b>Heart Disease Detected</b> 💔</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box" style="background-color: #C8E6C9;"><h3>✅ Prediction: <b>No Heart Disease</b> ❤️</h3></div>', unsafe_allow_html=True)

            # Visualization 1: Compare patient input to normal ranges
            st.markdown('<p class="subheader">Your Input vs. Normal Ranges</p>', unsafe_allow_html=True)
            features = ["Age", "Resting BP", "Cholesterol", "Max Heart Rate", "Oldpeak"]
            patient_values = [age, trestbps, chol, thalach, oldpeak]
            normal_ranges = [
                (30, 70),  # Age (typical range for adults)
                (90, 140), # Resting BP (normal range)
                (100, 200),# Cholesterol (desirable range)
                (100, 180),# Max Heart Rate (typical during exercise)
                (0, 2)     # Oldpeak (normal range)
            ]
            
            fig = go.Figure()
            for i, (feature, patient_val, (low, high)) in enumerate(zip(features, patient_values, normal_ranges)):
                fig.add_trace(go.Bar(
                    x=[feature],
                    y=[patient_val],
                    name="Your Value",
                    marker_color="#F44336" if patient_val < low or patient_val > high else "#4CAF50",
                    offsetgroup=i,
                    width=0.4
                ))
                fig.add_trace(go.Bar(
                    x=[feature],
                    y=[(low + high) / 2],
                    name="Normal Range",
                    marker_color="#2196F3",
                    opacity=0.5,
                    offsetgroup=i + 0.5,
                    width=0.4
                ))
                fig.add_shape(type="rect", x0=i-0.2, x1=i+0.2, y0=low, y1=high, fillcolor="#2196F3", opacity=0.2, layer="below")
            
            fig.update_layout(
                title="Your Values Compared to Normal Ranges",
                yaxis_title="Value",
                barmode="group",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Visualization 2: Feature importance (if supported)
            if hasattr(model, "feature_importances_"):
                st.markdown('<p class="subheader">Factors Influencing Your Prediction</p>', unsafe_allow_html=True)
                features = ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol", "Fasting BS", "Resting ECG", 
                            "Max Heart Rate", "Exercise Angina", "Oldpeak", "Slope", "Major Vessels", "Thalassemia"]
                importance = model.feature_importances_
                fig2 = px.bar(
                    x=features, y=importance, title="Feature Importance for Your Prediction",
                    labels={"x": "Feature", "y": "Importance"},
                    color=importance, color_continuous_scale="Viridis"
                )
                fig2.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

        except Exception as e:
            st.error(f"Error predicting: {e}")
