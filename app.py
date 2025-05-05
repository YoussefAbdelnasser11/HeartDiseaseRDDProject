import streamlit as st
from pyspark.sql import SparkSession
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

# Create Spark session
spark = SparkSession.builder.appName("HeartDiseaseApp").getOrCreate()
sc = spark.sparkContext

# Load and train model
@st.cache_resource
def load_model():
    data_path = "heart_cleveland_upload.csv"
    rdd = sc.textFile(data_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header)

    def safe_parse(line):
        parts = line.split(",")
        try:
            return [float(p) if p != "?" else None for p in parts]
        except:
            return None

    parsed = rdd.map(safe_parse).filter(lambda row: row is not None and None not in row)
    data = parsed.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:-1])))
    train, test = data.randomSplit([0.7, 0.3], seed=42)
    model = LogisticRegressionWithLBFGS.train(train)
    return model

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("Enter patient information to predict the risk of heart disease.")

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
    input_vector = Vectors.dense([inputs[key] for _, key in fields])
    prediction = model.predict(input_vector)
    if prediction == 1.0:
        st.error("⚠️ The model predicts: The patient **has** heart disease.")
    else:
        st.success("✅ The model predicts: The patient **does not** have heart disease.")

# Footer with your name and LinkedIn
st.markdown(
    "<hr style='margin-top: 50px;'>"
    "<div style='text-align: center;'>"
    "👨‍💻 Developed by <a href='https://www.linkedin.com/in/youssef-abdalnasser-33705b262/' target='_blank'>Youssef Abdelnasser</a>"
    "</div>",
    unsafe_allow_html=True
)
