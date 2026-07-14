import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetic Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling (Teal/Blue medical theme, glassmorphism card, neat form fields)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background and Card Container */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Title and Header styling */
    .main-title {
        color: #1A365D;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subtitle {
        color: #4A5568;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Custom Card for Results */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 1.5rem;
        color: white;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    .high-risk {
        background: linear-gradient(135deg, #e53e3e 0%, #b7791f 100%);
    }
    .low-risk {
        background: linear-gradient(135deg, #319795 0%, #2b6cb0 100%);
    }
    .risk-badge {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .risk-prob {
        font-size: 1.5rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    .risk-desc {
        font-size: 1.05rem;
        line-height: 1.5;
        border-top: 1px solid rgba(255,255,255,0.3);
        padding-top: 1rem;
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background-color: rgba(255,255,255,0.8);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 5px solid #319795;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to map UI diagnosis selection to category
def clean_diag(val):
    return val.split(' (')[0]

# Load ML components
@st.cache_resource
def load_model_and_preprocessor():
    models_dir = "models"
    model_path = os.path.join(models_dir, "best_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return None, None
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# App Layout
st.markdown("<h1 class='main-title'>🏥 Clinical Readmission Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predicting 30-Day Hospital Readmission for Diabetic Patients</p>", unsafe_allow_html=True)

if model is None or preprocessor is None:
    st.error("⚠️ Model or Preprocessor files not found. Please run the training pipeline first using `python run.py` to train and save the model.")
    st.info("Ensure that `models/best_model.joblib` and `models/preprocessor.joblib` are present in the project directory.")
    st.stop()

# Sidebar for metadata and system specs
with st.sidebar:
    st.markdown("### 🏥 System Overview")
    st.markdown("""
    <div class='sidebar-info'>
        <strong>Purpose:</strong> Predict readmission risk within 30 days of discharge.<br>
        <strong>Dataset:</strong> UCI Diabetes 130-US Hospitals (1999-2008).<br>
        <strong>Core Model:</strong> XGBoost / Random Forest Classifier.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Clinician Guidance")
    st.write("Patients flagged as **High Risk** should receive:")
    st.markdown("- Enhanced discharge planning\n- Detailed medication reconciliation\n- 48-hour follow-up phone call\n- Scheduled outpatient appointment within 7 days")
    
    st.markdown("---")
    st.markdown("**Version:** 1.0.0 (B.Tech Internship Project)")

# Create columns for layout
col_inputs, col_results = st.columns([2, 1.2])

with col_inputs:
    st.markdown("### 📋 Patient Clinical Record Entry")
    
    # Organize fields in tabs to avoid vertical crowding
    tab_demo, tab_clinical, tab_meds = st.tabs([
        "👤 Demographics", 
        "🏥 Clinical History", 
        "💊 Diabetes & Medications"
    ])
    
    with tab_demo:
        c1, c2 = st.columns(2)
        with c1:
            age_range = st.selectbox(
                "Age Group (Decade)",
                ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
                index=6
            )
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            race = st.selectbox("Race / Ethnicity", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"])
            medical_specialty = st.selectbox(
                "Admitting Specialty",
                ['InternalMedicine', 'Cardiology', 'FamilyPractice', 'GeneralPractice', 'Surgery-General', 'Orthopedics', 'Gastroenterology', 'Nephrology', 'Missing', 'Other']
            )

    with tab_clinical:
        c1, c2 = st.columns(2)
        with c1:
            time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 4)
            num_lab_procedures = st.slider("Number of Lab Procedures", 1, 120, 43)
            num_procedures = st.slider("Number of Non-Lab Procedures", 0, 6, 1)
            num_medications = st.slider("Number of Medications Prescribed", 1, 80, 16)
            number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 9)
        with c2:
            number_inpatient = st.slider("Prior Inpatient Admissions (Past Year)", 0, 20, 0)
            number_emergency = st.slider("Prior Emergency Room Visits (Past Year)", 0, 20, 0)
            number_outpatient = st.slider("Prior Outpatient Visits (Past Year)", 0, 20, 0)
            
            admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Other"])
            admission_source = st.selectbox("Admission Source", ["Emergency Room", "Referral", "Other"])
            discharge_disp = st.selectbox("Discharge Disposition", ["Home", "Transfer", "Other"])

    with tab_meds:
        # Grouped diagnosis dropdowns
        diag_categories = [
            "Circulatory (e.g. Heart disease, hypertension)",
            "Respiratory (e.g. Pneumonia, asthma, COPD)",
            "Digestive (e.g. Ulcers, gastroenteritis)",
            "Diabetes (ICD-9: 250)",
            "Injury (e.g. Fractures, poisoning)",
            "Musculoskeletal (e.g. Arthritis, back pain)",
            "Genitourinary (e.g. Kidney disease, UTI)",
            "Neoplasms (e.g. Cancers, tumors)",
            "Other (e.g. Infectious, mental health, skin)"
        ]
        
        diag_1_sel = st.selectbox("Primary Diagnosis Group (diag_1)", diag_categories, index=0)
        diag_2_sel = st.selectbox("Secondary Diagnosis Group (diag_2)", diag_categories, index=3)
        diag_3_sel = st.selectbox("Tertiary Diagnosis Group (diag_3)", diag_categories, index=8)
        
        c1, c2 = st.columns(2)
        with c1:
            max_glu_serum = st.selectbox("Max Glucose Serum Test", ["None", "Normal", ">200", ">300"])
            A1Cresult = st.selectbox("HbA1c Test Result", ["None", "Normal", ">7", ">8"])
            change = st.selectbox("Medication Dosage Change (Any)", ["No", "Ch"])
        with c2:
            diabetesMed = st.selectbox("Diabetic Medication Prescribed", ["Yes", "No"])
            metformin = st.selectbox("Metformin Treatment", ["No", "Steady", "Up", "Down"])
            insulin = st.selectbox("Insulin Treatment", ["No", "Steady", "Up", "Down"])

    # Clean the selected age range to extract numeric midpoint
    age_midpoint = 50
    try:
        age_str = age_range.replace('[', '').replace(')', '')
        start, end = map(int, age_str.split('-'))
        age_midpoint = (start + end) / 2
    except:
        pass

    # Build input DataFrame
    input_data = pd.DataFrame([{
        # Numeric columns
        'age': age_midpoint,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        
        # Categorical columns
        'race': race,
        'gender': gender,
        'medical_specialty': medical_specialty,
        'diag_1_cat': clean_diag(diag_1_sel),
        'diag_2_cat': clean_diag(diag_2_sel),
        'diag_3_cat': clean_diag(diag_3_sel),
        'admission_type_group': admission_type,
        'admission_source_group': admission_source,
        'discharge_disposition_group': discharge_disp,
        'max_glu_serum': max_glu_serum,
        'A1Cresult': A1Cresult,
        'change': change,
        'diabetesMed': diabetesMed,
        'metformin': metformin,
        'insulin': insulin
    }])

with col_results:
    st.markdown("### 🛡️ Decision Support Analysis")
    
    # Process and Predict
    processed_input = preprocessor.transform(input_data)
    
    # Map back to DataFrame to retain column names for model compatibility
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = cat_encoder.get_feature_names_out([
        'race', 'gender', 'medical_specialty', 'diag_1_cat', 'diag_2_cat', 'diag_3_cat',
        'admission_type_group', 'admission_source_group', 'discharge_disposition_group',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'metformin', 'insulin'
    ]).tolist()
    
    numeric_features = [
        'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ]
    
    processed_df = pd.DataFrame(processed_input, columns=numeric_features + encoded_cat_features)
    
    # Make Prediction
    pred = model.predict(processed_df)[0]
    prob = model.predict_proba(processed_df)[0][1]
    
    # Display Card based on Risk
    if pred == 1 or prob >= 0.5:
        st.markdown(f"""
        <div class="result-card high-risk">
            <h2>⚠️ PREDICTION</h2>
            <div class="risk-badge">HIGH RISK</div>
            <div class="risk-prob">Probability: {prob*100:.1f}%</div>
            <div class="risk-desc">
                <strong>Attention:</strong> Patient has a high likelihood of readmission within 30 days.<br>
                Please review medication regimes, schedule early post-discharge follow-up, and optimize the discharge plan.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card low-risk">
            <h2>✔️ PREDICTION</h2>
            <div class="risk-badge">LOW RISK</div>
            <div class="risk-prob">Probability: {prob*100:.1f}%</div>
            <div class="risk-desc">
                Patient is classified as low risk for 30-day hospital readmission.<br>
                Apply standard institutional discharge protocols.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📊 Top Risk Indicators")
    
    # Display patient metrics visualization
    # We can plot a comparison of key numeric metrics for the patient against their general scale to show what's driving the prediction
    risk_factors = {
        'Inpatient Visits': (number_inpatient, 4, 'inpatient admissions'),
        'Emergency Visits': (number_emergency, 2, 'emergency room visits'),
        'Hospital Stay (Days)': (time_in_hospital, 7, 'days in hospital'),
        'Medications Count': (num_medications, 25, 'different drugs')
    }
    
    # Render indicators with status colors
    for name, (val, threshold, unit) in risk_factors.items():
        is_high = val >= threshold
        color = "red" if is_high else "green"
        indicator = "🔴" if is_high else "🟢"
        st.write(f"{indicator} **{name}:** {val} {unit} (Threshold for high risk: {threshold})")
        
    st.markdown("---")
    st.caption("Disclaimer: This tool is an AI-based clinical decision support assistant. Final clinical judgements must be made by qualified healthcare professionals.")
