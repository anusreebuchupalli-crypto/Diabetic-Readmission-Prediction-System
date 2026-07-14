import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

def map_age(age_str):
    # Maps interval strings like '[0-10)' to their midpoints
    try:
        age_str = age_str.replace('[', '').replace(')', '')
        start, end = map(int, age_str.split('-'))
        return (start + end) / 2
    except:
        return 50 # Default median age if parsing fails

def map_icd9_to_category(code):
    if pd.isna(code) or code == '?':
        return 'Missing'
    
    code = str(code).strip()
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    
    try:
        # Convert to numeric float for ranges
        val = float(code)
        if 390 <= val <= 459 or val == 785:
            return 'Circulatory'
        elif 460 <= val <= 519 or val == 786:
            return 'Respiratory'
        elif 520 <= val <= 579 or val == 787:
            return 'Digestive'
        elif 250 <= val < 251: # Diabetes is 250.xx
            return 'Diabetes'
        elif 800 <= val <= 999:
            return 'Injury'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif 580 <= val <= 629 or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'

def preprocess_data():
    raw_data_path = os.path.join("dataset", "raw", "dataset_diabetes", "diabetic_data.csv")
    if not os.path.exists(raw_data_path):
        # Check if it was extracted directly to raw/
        raw_data_path = os.path.join("dataset", "raw", "diabetic_data.csv")
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError("diabetic_data.csv not found. Run download_dataset.py first.")

    print(f"Loading raw dataset from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    print(f"Original shape: {df.shape}")

    # 1. Replace '?' with NaN
    df = df.replace('?', np.nan)

    # 2. Exclude patients who expired or were discharged to hospice (cannot be readmitted)
    # discharge_disposition_id maps:
    # 11: Expired, 13: Hospice/home, 14: Hospice/medical facility,
    # 19: Expired at home, 20: Expired in medical facility, 21: Expired place unknown,
    # 25: Expired (additional code)
    dead_hospice_ids = [11, 13, 14, 19, 20, 21, 25]
    df = df[~df['discharge_disposition_id'].isin(dead_hospice_ids)]
    print(f"Shape after removing deceased/hospice patients: {df.shape}")

    # 3. Keep only the first encounter per patient to prevent data leakage
    df = df.sort_values('encounter_id')
    df = df.drop_duplicates(subset='patient_nbr', keep='first')
    print(f"Shape after keeping first encounter per patient: {df.shape}")

    # 4. Handle target variable 'readmitted'
    # Classify readmissions: '<30' as 1, others ('>30' and 'NO') as 0
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    print(f"Target variable distribution:\n{df['readmitted'].value_counts(normalize=True)}")

    # 5. Drop columns with extremely high missingness or uninformative values
    # 'weight' is ~97% missing, 'payer_code' is ~40% missing and not clinically relevant.
    # 'encounter_id' and 'patient_nbr' are IDs.
    cols_to_drop = ['weight', 'payer_code', 'encounter_id', 'patient_nbr']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 6. Feature Engineering
    # Map age interval to numeric midpoint
    df['age'] = df['age'].apply(map_age)

    # Clean gender: remove 'Unknown/Invalid' rows (very small count)
    df = df[df['gender'].isin(['Male', 'Female'])]
    
    # Map diagnoses to broad ICD-9 categories
    df['diag_1_cat'] = df['diag_1'].apply(map_icd9_to_category)
    df['diag_2_cat'] = df['diag_2'].apply(map_icd9_to_category)
    df['diag_3_cat'] = df['diag_3'].apply(map_icd9_to_category)
    
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

    # Medical specialty: keep top 10 and map rest to 'Other', fill missing with 'Missing'
    df['medical_specialty'] = df['medical_specialty'].fillna('Missing')
    top_specialties = df['medical_specialty'].value_counts().index[:10]
    df['medical_specialty'] = df['medical_specialty'].apply(lambda x: x if x in top_specialties else 'Other')

    # Group admission IDs to reduce cardinality
    # Admission type: 1, 2, 7 -> Emergency/Urgent; 3 -> Elective; others -> Other
    df['admission_type_group'] = df['admission_type_id'].apply(
        lambda x: 'Emergency' if x in [1, 2, 7] else ('Elective' if x == 3 else 'Other')
    )
    # Admission source: 7 -> Emergency Room; 1, 2 -> Referral; others -> Other
    df['admission_source_group'] = df['admission_source_id'].apply(
        lambda x: 'Emergency Room' if x == 7 else ('Referral' if x in [1, 2] else 'Other')
    )
    # Discharge disposition: 1 -> Home; 3, 4, 5, 22 -> Transfer to other facility; others -> Other
    df['discharge_disposition_group'] = df['discharge_disposition_id'].apply(
        lambda x: 'Home' if x == 1 else ('Transfer' if x in [3, 4, 5, 22] else 'Other')
    )

    df = df.drop(columns=['admission_type_id', 'admission_source_id', 'discharge_disposition_id'])

    # We will select key features to keep model clean and fast.
    # Predictors:
    # Numeric: age, time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses
    # Categorical: race, gender, medical_specialty, diag_1_cat, diag_2_cat, diag_3_cat, admission_type_group, admission_source_group, discharge_disposition_group,
    #              max_glu_serum, A1Cresult, change, diabetesMed, metformin, insulin
    
    # We will only keep top 2 medications (metformin, insulin) to simplify the feature space and Streamlit input form.
    meds_to_keep = ['metformin', 'insulin']
    all_meds = ['repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 
                'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 
                'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    df = df.drop(columns=[m for m in all_meds if m in df.columns], errors='ignore')

    # Define feature lists
    numeric_features = [
        'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ]
    categorical_features = [
        'race', 'gender', 'medical_specialty', 'diag_1_cat', 'diag_2_cat', 'diag_3_cat',
        'admission_type_group', 'admission_source_group', 'discharge_disposition_group',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'metformin', 'insulin'
    ]

    target = 'readmitted'
    
    X = df[numeric_features + categorical_features]
    y = df[target]

    # Fill remaining missing categoricals with 'Missing'
    for col in categorical_features:
        X[col] = X[col].fillna('Missing')

    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Creating preprocessing pipelines...")
    # Preprocessor pipeline
    numeric_transformer = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    
    categorical_transformer = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [i for i in range(len(numeric_features), len(X.columns))])
        ],
        remainder='passthrough'
    )

    # Let's use ColumnTransformer directly on original columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + encoded_cat_features

    # Convert processed data back to DataFrames to retain column names
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    print(f"Shape after preprocessing: X_train={X_train_df.shape}, X_test={X_test_df.shape}")

    # Apply SMOTE to handle class imbalance
    print("Applying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_df, y_train)
    print(f"Resampled target distribution:\n{y_train_res.value_counts()}")
    print(f"Shape after SMOTE: X_train_res={X_train_res.shape}")

    # Create directories for saving processed data
    processed_dir = os.path.join("dataset", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Save training and test splits
    print("Saving processed datasets...")
    X_train_res.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    y_train_res.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    # Also save raw train/test sets to fit shap / run predictions inside streamlit later if needed
    X_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    # Save fitted preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()
