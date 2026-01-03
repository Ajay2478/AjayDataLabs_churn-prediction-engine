# 🛡️ AjayDataLabs: Retention Savior Engine
### *Professional Churn Prediction System*

### 🚀 Business Objective
Customer acquisition costs **5x more** than retention. This engine predicts which customers are about to leave (churn) and explains *exactly why*, enabling retention teams to intervene proactively.

**Value Proposition:**
* Identifies high-risk customers with **74% Recall** (capturing 3 out of 4 churners).
* Provides explainable AI insights (SHAP) to guide marketing offers.
* Reduces revenue leakage by prioritizing at-risk accounts.

---

### 🧠 The Solution
A machine learning pipeline built on **XGBoost** and **SMOTE**, deployed via a **Streamlit** dashboard for real-time risk assessment.

---

### ⚙️ Technical Architecture

| Component | Tech Stack | Role |
| :--- | :--- | :--- |
| **Model** | XGBoost | High-performance gradient boosting for tabular data. |
| **Imbalance Handling** | SMOTE | Synthetic minority oversampling to fix the 26% churn imbalance. |
| **Explainability** | SHAP | Global & local feature importance (Why did *this* user churn?). |
| **App** | Streamlit | Interactive web interface for business users. |
| **Processing** | Pandas/Scikit-Learn | Automated ETL pipeline with scaling & encoding. |

---

### 🔍 Key Insights (Data-Driven)
During the analysis of 7,000+ customers, the model discovered:
1.  **Fiber Optic Risk:** Customers with Fiber Optic internet are the highest churn risk (likely due to price/service mismatch).
2.  **Contract Sensitivity:** "Month-to-month" contracts are the #1 predictor of churn.
3.  **Tenure Factor:** Churn probability drops significantly after 24 months of tenure.

---