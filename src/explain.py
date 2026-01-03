import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'churn_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_xgb.pkl')
IMG_PATH = os.path.join(BASE_DIR, 'app', 'shap_summary.png')

print("🚀 Starting SHAP Explanation...")

# 2. Load Data & Model
df = pd.read_csv(DATA_PATH)
X = df.drop('Churn', axis=1)
model = joblib.load(MODEL_PATH)

# 3. Calculate SHAP Values
# We use a random sample of 1000 rows to speed it up (calculating on all 7000 takes too long)
print("🧮 Calculating SHAP values (this might take 10-20 seconds)...")
X_sample = X.sample(n=1000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# 4. Generate & Save Summary Plot
print("🎨 Generating Summary Plot...")
plt.figure(figsize=(10, 6))
# Create the plot but don't show it yet
shap.summary_plot(shap_values, X_sample, show=False)

# Save clearly for the app
os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
plt.savefig(IMG_PATH, bbox_inches='tight', dpi=300)
plt.close()

print(f"✅ SHAP Summary saved to: {IMG_PATH}")
print("   (Check this image to see what drives churn!)")