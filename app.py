import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="🔍 Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown("---")

st.header("📝 Enter Patient Data")
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Prepare input
sex_val = 1 if sex == "Male" else 0
input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Load scaler and model
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("random_forest_model.pkl")

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)[0]
    proba = model.predict_proba(input_data_scaled)[0]
    
    st.subheader("📊 Prediction Result")
    result_text = "✅ No Heart Disease Detected" if prediction == 0 else "⚠️ High Risk of Heart Disease"
    result_color = "green" if prediction == 0 else "red"
    st.markdown(f"<h3 style='color:{result_color}'>{result_text}</h3>", unsafe_allow_html=True)
    
    # Pie chart for prediction probability
    st.markdown("#### Prediction Probability")
    fig1, ax1 = plt.subplots()
    ax1.pie(proba, labels=["No Disease", "Heart Disease"], autopct="%1.1f%%", startangle=90, colors=["#66bb6a", "#ef5350"])
    ax1.axis("equal")
    st.pyplot(fig1)
    
    # Radar Chart: normalize values against max scale
    st.markdown("#### Patient Profile vs Typical Ranges")
    features = ['Age', 'BP', 'Chol', 'HR', 'Oldpeak']
    values = [age/100, trestbps/200, chol/600, thalach/220, oldpeak/6]
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig2, ax2 = plt.subplots(subplot_kw={"polar": True})
    ax2.plot(angles, values, "o-", linewidth=2)
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_thetagrids(np.degrees(angles[:-1]), features)
    ax2.set_title("Patient's Vitals Radar Chart")
    st.pyplot(fig2)

    # Add creator's name at the bottom
    st.markdown("---")
    st.markdown("### Created by [Youssef Abdelnasser](https://www.linkedin.com/in/youssef-abdalnasser-33705b262/)")

except Exception as e:
    st.error("Model or scaler file not found, or prediction failed.")
    st.text(str(e))
