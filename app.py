import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="🔍 Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown("---")

st.header("📝 Enter Patient Data")
age = st.slider("Age", 20, 100, 50, help="Patient's age in years")
sex = st.selectbox("Sex", ["Male", "Female"], help="Patient's gender")
cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-Anginal Pain (2)", "Asymptomatic (3)"],
                  help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120, help="Resting blood pressure in mm Hg")
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240, help="Serum cholesterol in mg/dl")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"],
                   help="0: Fasting blood sugar ≤ 120 mg/dl, 1: > 120 mg/dl")
restecg = st.selectbox("Resting ECG Results", ["Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)"],
                       help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
thalach = st.slider("Maximum Heart Rate Achieved", 70, 220, 150, help="Maximum heart rate during exercise")
exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"],
                     help="0: No angina during exercise, 1: Yes")
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest")
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"],
                     help="0: Upsloping, 1: Flat, 2: Downsloping")
ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3, 4],
                  help="Number of major vessels (0-4) colored by fluoroscopy")
thal = st.selectbox("Thalassemia", ["No Data (0)", "Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"],
                    help="0: No data, 1: Normal, 2: Fixed defect, 3: Reversible defect")

# Convert selectbox inputs to numeric values
def extract_numeric(value):
    # Extract the number from strings like "No (0)" or "Atypical Angina (1)"
    return int(value.split('(')[1].strip(')'))

sex_val = 1 if sex == "Male" else 0
cp_val = extract_numeric(cp)
fbs_val = extract_numeric(fbs)
restecg_val = extract_numeric(restecg)
exang_val = extract_numeric(exang)
slope_val = extract_numeric(slope)
thal_val = extract_numeric(thal)

# Prepare one-hot encoded variables
cp_1 = 1 if cp_val == 1 else 0
cp_2 = 1 if cp_val == 2 else 0
cp_3 = 1 if cp_val == 3 else 0
restecg_1 = 1 if restecg_val == 1 else 0
restecg_2 = 1 if restecg_val == 2 else 0
slope_1 = 1 if slope_val == 1 else 0
slope_2 = 1 if slope_val == 2 else 0
thal_1 = 1 if thal_val == 1 else 0
thal_2 = 1 if thal_val == 2 else 0
thal_3 = 1 if thal_val == 3 else 0

# Create input array with 19 features
input_data = np.array([[age, sex_val, trestbps, chol, fbs_val, restecg_1, restecg_2, thalach,
                        exang_val, oldpeak, slope_1, slope_2, ca, cp_1, cp_2, cp_3, thal_1, thal_2, thal_3]])

# Define all possible features (matching training feature names with .0 suffix)
all_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg_1.0', 'restecg_2.0', 'thalach',
                'exang', 'oldpeak', 'slope_1.0', 'slope_2.0', 'ca', 'cp_1.0', 'cp_2.0', 'cp_3.0', 'thal_1.0', 'thal_2.0', 'thal_3.0']

# Load model
try:
    model = joblib.load("random_forest_model.pkl(3)")

    # Validate input data
    input_df = pd.DataFrame(input_data, columns=all_features)

    # Ensure numeric data
    input_df = input_df.astype(float)

    # Predict directly (no scaling, using all 19 features)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
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
    st.error("Model file not found, or prediction failed.")
    st.text(str(e))
