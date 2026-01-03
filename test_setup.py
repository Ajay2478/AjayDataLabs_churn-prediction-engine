print("1. Starting Test...")
import pandas as pd
print(f"2. Pandas Imported Successfully (Version: {pd.__version__})")

try:
    df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"3. Data Loaded Successfully! Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")

print("4. Test Complete.")