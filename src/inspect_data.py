import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\data\raw\loans_granting_model_dataset.csv"

print("--- 1. Loading Dataset Overview ---")
df = pd.read_csv(DATA_PATH)
print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")

print("\n--- 2. Columns & Data Types ---")
info_df = pd.DataFrame({
    'Column': df.columns,
    'Dtype': df.dtypes.values,
    'Non-Null Count': df.notnull().sum().values,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
    'Unique Values': [df[col].nunique() for col in df.columns]
})
print(info_df.to_string(index=False))

print("\n--- 3. Target (Default) Distribution ---")
if 'Default' in df.columns:
    val_counts = df['Default'].value_counts(dropna=False)
    val_pct = df['Default'].value_counts(normalize=True, dropna=False) * 100
    for val in val_counts.index:
        label = "Default (1)" if val == 1 else ("Non-Default / Fully Paid (0)" if val == 0 else str(val))
        print(f"  {label}: {val_counts[val]:,} ({val_pct[val]:.2f}%)")

print("\n--- 4. Numerical Summary Statistics ---")
print(df.describe().round(2).to_string())

print("\n--- 5. Categorical Features Unique Values / Examples ---")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    sample_vals = df[col].dropna().unique()[:8]
    print(f"  {col} ({df[col].nunique()} unique): {sample_vals}")

print("\n--- 6. Duplicate Rows Check ---")
dup_count = df.duplicated().sum()
print(f"Duplicate full rows: {dup_count}")
if 'id' in df.columns:
    print(f"Duplicate IDs: {df['id'].duplicated().sum()}")
