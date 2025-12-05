import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

st.set_page_config(page_title="Heart Disease Risk Classifier", layout="centered")

st.title("Heart Disease Risk Classifier")
st.markdown("Enter patient data on the left and click **Predict**.")

# ------------- Load model -------------
MODEL_PATH = "heart_disease_xgb_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.stop()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return joblib.load(path)

model = load_model()

# ------------- Input widgets -------------
st.sidebar.header("Patient information")

# Numeric inputs
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=55)
trestbps = st.sidebar.number_input("Resting blood pressure (trestbps)", min_value=50, max_value=300, value=130)
chol = st.sidebar.number_input("Cholesterol (chol)", min_value=50, max_value=1000, value=200)
thalch = st.sidebar.number_input("Max heart rate achieved (thalch)", min_value=20, max_value=250, value=150)
oldpeak = st.sidebar.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Categorical inputs (use the textual categories present in your dataset)
sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
cp = st.sidebar.selectbox("Chest pain type (cp)", options=[
    "typical angina",
    "atypical angina",
    "non-anginal",
    "asymptomatic"
])
restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[
    "normal",
    "lv hypertrophy"
])
# If your pipeline expects fbs/exang include them; otherwise you can omit.
fbs = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=["True", "False"])
exang = st.sidebar.selectbox("Exercise induced angina (exang)", options=["True", "False"])

# ------------- Prepare input dataframe -------------
# Ensure column names & order match how the pipeline expects them.
input_dict = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalch": thalch,
    "oldpeak": oldpeak,
    "sex": sex,
    "cp": cp,
    "restecg": restecg,
    "fbs": True if fbs == "True" else False,
    "exang": True if exang == "True" else False
}

# If your pipeline was trained with only a subset of categorical features, remove the extras accordingly.
X = pd.DataFrame([input_dict])

st.subheader("Input preview")
st.dataframe(X.T, height=300)

# ------------- Prediction -------------
if st.button("Predict"):
    try:
        # Pipeline should accept raw X (it will preprocess internally)
        pred_proba = model.predict_proba(X)[:, 1]  # probability of class 1
        pred = model.predict(X)[0]
    except Exception as e:
        st.error("Prediction failed. Check that the app input columns match the pipeline's expected columns.")
        st.exception(e)
    else:
        risk_pct = float(pred_proba[0]) * 100
        st.metric(label="Predicted Risk (%)", value=f"{risk_pct:.1f}%")
        st.write("Predicted class:", "Heart disease" if pred == 1 else "No heart disease")

        # Show brief explanation
        st.markdown("""
        **Notes**
        - Model pipeline handles encoding and scaling internally.
        - If you included fewer/more columns when you trained the pipeline, update the app inputs to match exactly.
        """)

        # Optional: show SHAP (if shap installed and model supports it)
        if st.checkbox("Show SHAP explanation (may be slow)"):
            try:
                import shap
                st.info("Computing SHAP values — this may take a few seconds.")
                explainer = shap.Explainer(model.named_steps['model'])
                # Get preprocessed feature matrix to pass to explainer (if needed)
                # This simple approach calls preprocess step then model
                preproc = model.named_steps['preprocessor']
                X_trans = preproc.transform(X)
                shap_values = explainer(X_trans)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.plots.waterfall(shap_values[0], show=False)
                import matplotlib.pyplot as plt
                plt.tight_layout()
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error("SHAP explanation failed (model type / shap compatibility).")
                st.exception(e)

# ------------- Footer -------------
st.markdown("---")
st.markdown("Developed by Yash Patil")