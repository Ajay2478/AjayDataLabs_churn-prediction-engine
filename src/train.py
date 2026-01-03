import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'churn_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_xgb.pkl')

print("🚀 Starting Model Training...")

# 2. Load Data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Processed data not found at {DATA_PATH}. Run src/preprocess.py first.")

df = pd.read_csv(DATA_PATH)

# 3. Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']

# 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Original Train Shape: {X_train.shape}")
print(f"   Class Distribution (Train): \n{y_train.value_counts(normalize=True)}")

# 4. Handle Imbalance (SMOTE)
print("⚖️  Applying SMOTE to fix class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"   New Train Shape (after SMOTE): {X_train_resampled.shape}")
print(f"   New Class Distribution: \n{y_train_resampled.value_counts(normalize=True)}")

# 5. Train XGBoost
print("🧠 Training XGBoost Model...")
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)

# 6. Evaluate
print("📉 Evaluating...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred) 
auc = roc_auc_score(y_test, y_prob)

print(f"\n🏆 --- Final Metrics ---")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}  <-- (Target: > 0.75)")
print(f"   ROC-AUC:   {auc:.4f}")
print("\n   Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save Model
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")