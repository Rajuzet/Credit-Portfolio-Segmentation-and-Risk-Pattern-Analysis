import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\data\raw\loans_granting_model_dataset.csv"
PROCESSED_DIR = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\data\processed"
MODELS_DIR = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_preprocess(sample_size=50000, random_state=42):
    print("Step 3.1: Reading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"Loaded raw data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    print("Step 3.2: Cleaning & Parsing Dates...")
    # Convert issue_d to datetime
    df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['issue_year'] = df['issue_date'].dt.year.fillna(2015).astype(int)
    df['issue_month'] = df['issue_date'].dt.month.fillna(6).astype(int)

    print("Step 3.3: Mapping Employment Length...")
    emp_map = {
        '10+ years': 10,
        '9 years': 9,
        '8 years': 8,
        '7 years': 7,
        '6 years': 6,
        '5 years': 5,
        '4 years': 4,
        '3 years': 3,
        '2 years': 2,
        '1 year': 1,
        '< 1 year': 0,
        'NI': 0
    }
    df['emp_length_years'] = df['emp_length'].map(emp_map).fillna(0).astype(int)

    print("Step 3.4: Handling Outliers & Skewness...")
    # Cap DTI at 60 (removes nonsensical extreme values like 999 while retaining 99.9% of distribution)
    df['dti_capped'] = df['dti_n'].clip(lower=0.0, upper=60.0)
    
    # Log transforms for skewed monetary variables
    df['log_revenue'] = np.log1p(df['revenue'].clip(lower=1000.0))
    df['log_loan_amnt'] = np.log1p(df['loan_amnt'])

    print("Step 3.5: Encoding Categoricals...")
    # Categorical columns to encode
    home_dummies = pd.get_dummies(df['home_ownership_n'], prefix='home', drop_first=False, dtype=int)
    purpose_dummies = pd.get_dummies(df['purpose'], prefix='purpose', drop_first=False, dtype=int)

    # Feature columns dedicated exclusively for unsupervised clustering
    feature_cols_numerical = [
        'log_revenue',
        'log_loan_amnt',
        'fico_n',
        'dti_capped',
        'emp_length_years',
        'experience_c'
    ]
    clustering_feature_cols = feature_cols_numerical + list(home_dummies.columns) + list(purpose_dummies.columns)
    
    # Combine dummies into full dataframe
    df_full = pd.concat([df, home_dummies, purpose_dummies], axis=1)

    print(f"Total clustering features: {len(clustering_feature_cols)}")
    print(f"Features: {clustering_feature_cols}")

    print("Step 3.6: Creating Stratified Sample for High-Performance Clustering...")
    # Sample 50,000 records stratified by 'Default' for reproducible, memory-safe clustering & manifold learning
    if sample_size and sample_size < len(df_full):
        df_sample, _ = train_test_split(
            df_full,
            train_size=sample_size,
            stratify=df_full['Default'],
            random_state=random_state
        )
    else:
        df_sample = df_full.copy()
        
    df_sample = df_sample.reset_index(drop=True)
    print(f"Sample size: {len(df_sample):,} rows. Default distribution in sample:")
    print(df_sample['Default'].value_counts(normalize=True).round(4) * 100)

    print("Step 3.7: Fitting Standard Scaler on Clustering Features...")
    scaler = StandardScaler()
    X_sample_scaled = scaler.fit_transform(df_sample[clustering_feature_cols])

    # Save fitted scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    # Save metadata
    metadata = {
        "clustering_feature_cols": clustering_feature_cols,
        "feature_cols_numerical": feature_cols_numerical,
        "home_cols": list(home_dummies.columns),
        "purpose_cols": list(purpose_dummies.columns),
        "sample_size": len(df_sample),
        "total_records": len(df_full),
        "random_state": random_state
    }
    metadata_path = os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")

    print("Step 3.8: Saving Cleaned Sample Data...")
    sample_parquet_path = os.path.join(PROCESSED_DIR, "cleaned_loans_sample.parquet")
    sample_csv_path = os.path.join(PROCESSED_DIR, "cleaned_loans_sample.csv")
    
    # Select essential columns to keep the file lean and fast
    essential_cols = [
        'id', 'issue_d', 'issue_year', 'issue_month',
        'revenue', 'log_revenue',
        'loan_amnt', 'log_loan_amnt',
        'fico_n', 'dti_n', 'dti_capped',
        'emp_length', 'emp_length_years', 'experience_c',
        'home_ownership_n', 'purpose', 'addr_state',
        'Default'
    ] + list(home_dummies.columns) + list(purpose_dummies.columns)

    df_sample_clean = df_sample[essential_cols]
    df_sample_clean.to_parquet(sample_parquet_path, index=False)
    # Save a CSV copy as well
    df_sample_clean.to_csv(sample_csv_path, index=False)
    print(f"Saved sample parquet ({os.path.getsize(sample_parquet_path) / (1024*1024):.2f} MB) and csv to {PROCESSED_DIR}")

    # Also save the scaled matrix directly
    scaled_matrix_path = os.path.join(PROCESSED_DIR, "X_sample_scaled.npy")
    np.save(scaled_matrix_path, X_sample_scaled)
    print(f"Saved scaled feature matrix (shape: {X_sample_scaled.shape}) to {scaled_matrix_path}")

    print("Step 3: Preprocessing & Feature Engineering Completed Successfully!")

if __name__ == "__main__":
    load_and_preprocess()
