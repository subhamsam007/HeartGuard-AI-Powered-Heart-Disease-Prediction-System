import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
            
<style>
    
            
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E74C3C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .high-risk {
        background-color: #FFE6E6;
        color: #D63031;
        border: 2px solid #D63031;
    }
    .low-risk {
        background-color: #E6F7E6;
        color: #27AE60;
        border: 2px solid #27AE60;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3498DB;
        margin-bottom: 1rem;
    }
    .input-group {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    body, .stApp {
        background: linear-gradient(135deg, #000000 0%, #0a0033 25%, #001a4d 50%, #330033 75%, #660000 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load saved model, scaler, and expected columns
model = joblib.load(".venv/models/KNN_heart.pkl")
scaler = joblib.load(".venv/models/scaler.pkl")
expected_columns = joblib.load(".venv/models/columns.pkl")

# Main content
st.markdown('<div class="main-header">🫀 Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter your health details below to assess your risk of heart disease</div>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<div class="sidebar-header">📋 Health Information</div>', unsafe_allow_html=True)
    
    # Personal Information
    st.markdown("### 👤 Personal Details")
    age = st.slider("Age", 18, 100, 40, help="Your age in years")
    sex = st.selectbox("Sex", ["M", "F"], help="Biological sex")
    
    # Medical Measurements
    st.markdown("### 🩺 Medical Measurements")
    col1, col2 = st.columns(2)
    with col1:
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure at rest")
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, help="Total cholesterol level")
    with col2:
        max_hr = st.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved")
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise")
    
    # Categorical Inputs
    st.markdown("### 📊 Categorical Data")
    col3, col4 = st.columns(2)
    with col3:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], 
                                  help="Type of chest pain experienced")
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], 
                                  format_func=lambda x: "Yes" if x else "No",
                                  help="Is fasting blood sugar > 120 mg/dL?")
    with col4:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], 
                                   help="Resting electrocardiogram results")
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], 
                                       format_func=lambda x: "Yes" if x == "Y" else "No",
                                       help="Angina induced by exercise")
    
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], 
                            help="Slope of the peak exercise ST segment")

# Prediction button in main area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Predict Heart Disease Risk", key="predict")

# Prediction logic
if predict_button:
    with st.spinner("Analyzing your health data..."):
        # Create a raw input dictionary
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Show result
        if prediction == 1:
            st.markdown('<div class="prediction-result high-risk">⚠️ High Risk of Heart Disease</div>', unsafe_allow_html=True)
            st.warning("**Recommendation:** Please consult with a healthcare professional for a comprehensive evaluation.")
        else:
            st.markdown('<div class="prediction-result low-risk">✅ Low Risk of Heart Disease</div>', unsafe_allow_html=True)
            st.success("**Great news!** Your inputs suggest a low risk, but always maintain a healthy lifestyle.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; font-size: 0.8rem;">
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. Always consult a qualified healthcare provider for medical concerns.</p>
    <h4>Built with ❤️ by Subham</h4>
</div>
""", unsafe_allow_html=True)


