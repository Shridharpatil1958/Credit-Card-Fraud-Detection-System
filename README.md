# Credit Card Fraud Detection — Full Project

An end-to-end imbalanced-classification project: EDA → resampling (SMOTE/ADASYN/undersampling)
→ model comparison (Logistic Regression, XGBoost, Isolation Forest) → cost-based threshold
tuning → SHAP explainability → an interactive Streamlit demo.

## Files

```
fraud_detection/
├── README.md                          <- this file
├── data/
│   ├── generate_data.py               <- builds the synthetic dataset
│   └── creditcard.csv                 <- generated data (Time, V1-V28, Amount, Class)
├── fraud_detection_project.ipynb      <- the full analysis notebook (already executed)
├── models/                            <- saved model artifacts (from notebook Section 9)
│   ├── fraud_xgb_model.pkl
│   ├── scaler.pkl
│   └── model_config.pkl
├── outputs/                           <- exported PNG charts from the notebook
└── app/
    └── streamlit_app.py               <- interactive fraud-scoring demo
```

## About the dataset

This sandbox can't reach kaggle.com, so `data/generate_data.py` builds a **synthetic** dataset
that mirrors the real one's structure and difficulty:
- 60,000 transactions, ~0.2% fraud (matches the real ~0.17%)
- Columns: `Time`, `V1`-`V28` (PCA-like anonymized features), `Amount`, `Class`
- Realistic touches: fraud skews toward late-night hours and toward either very small
  ("card testing") or very large amounts; some label noise; overlapping (not perfectly
  separable) classes

**To use the real data:** download `creditcard.csv` from
[kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
drop it into `data/`, and re-run the notebook — every downstream cell works unchanged since the
column schema matches.

## How to run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap streamlit matplotlib seaborn joblib

# 2. (Optional) regenerate the dataset
python data/generate_data.py

# 3. Open and run the notebook (already executed once, so outputs/models exist)
jupyter notebook fraud_detection_project.ipynb

# 4. Launch the interactive demo
streamlit run app/streamlit_app.py
```

## What the notebook covers

1. **EDA** — class imbalance, amount/time distributions, feature correlations
2. **Train/test split** — done *before* resampling, to avoid leakage
3. **Imbalance handling** — Random Undersampling, SMOTE, ADASYN, SMOTE+Tomek, and
   class-weighting (no resampling) compared side by side
4. **Models** — Logistic Regression + XGBoost on each resampled set, XGBoost with
   `scale_pos_weight`, and an unsupervised Isolation Forest baseline
5. **Evaluation** — Precision-Recall AUC (not accuracy!), ROC-AUC, F1, full comparison table
6. **Cost-based threshold tuning** — sweeps the decision threshold against an explicit
   false-negative vs. false-positive cost model to find the business-optimal cutoff
   (rather than a generic 0.5)
7. **SHAP explainability** — global feature importance + a single flagged-transaction
   explanation, the kind a fraud analyst would actually need to see
8. Saves the final model/scaler/threshold for the Streamlit app

## Key takeaway

Accuracy is a trap at 0.2% fraud — a model that predicts "never fraud" scores ~99.8% accuracy
and is useless. Everything here is built around metrics and decisions (PR-AUC, cost-optimal
threshold) that actually matter for a real fraud system.
