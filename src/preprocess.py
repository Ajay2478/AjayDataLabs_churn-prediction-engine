import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'churn_processed.csv')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

print("🚀 Starting Data Processing...")

# 2. Load Data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Cannot find data at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"   Original Shape: {df.shape}")

# 3. CLEANING: Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# FIXED: Replaced inplace=True with direct assignment to silence warning
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# 4. ENCODING: Binary
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# 5. ENCODING: One-Hot
cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaymentMethod']
df_processed = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 6. SCALING
scaler = MinMaxScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

# 7. SAVE (Robust)
# FIXED: Create directories if they don't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

df_processed.drop('customerID', axis=1, inplace=True)
df_processed.to_csv(OUTPUT_PATH, index=False)
joblib.dump(scaler, SCALER_PATH)

print(f"✅ SUCCESS! Processed data saved to: {OUTPUT_PATH}")
print(f"📊 Final Column Count: {df_processed.shape[1]}")