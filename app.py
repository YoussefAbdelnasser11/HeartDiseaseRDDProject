import streamlit as st
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("🔍 Heart Disease Prediction")
st.markdown(
    "Created by [Youssef Abdelnasser](https://www.linkedin.com/in/youssef-abdalnasser-33705b262/)",
    unsafe_allow_html=True,
)

st.header("Enter Patient Data")

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
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    try:
        model = joblib.load("model_link.txt")  # You must upload this model to your environment or repo
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
        st.success("Prediction: " + ("Heart Disease Detected 💔" if prediction[0] == 1 else "No Heart Disease ❤️"))
    except Exception as e:
        st.error(f"Error loading model or predicting: {e}")
