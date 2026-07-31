"""
Fraud Detection — interactive demo app.

Run with:
    streamlit run app/streamlit_app.py

Loads the model trained in fraud_detection_project.ipynb (Section 9) and lets you:
  1. Score a random test-set transaction, or enter custom feature values
  2. See the fraud probability vs. the cost-optimal decision threshold
  3. See a SHAP explanation of *why* the model flagged (or cleared) it
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_xgb_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    config = joblib.load(os.path.join(BASE_DIR, "models", "model_config.pkl"))
    explainer = shap.TreeExplainer(model)
    return model, scaler, config, explainer


@st.cache_data
def load_test_sample(n=300):
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "creditcard.csv"))
    # bias the sample toward containing at least a few real fraud cases to explore
    fraud = df[df.Class == 1].sample(min(15, (df.Class == 1).sum()), random_state=1)
    legit = df[df.Class == 0].sample(n - len(fraud), random_state=1)
    return pd.concat([fraud, legit]).sample(frac=1, random_state=1).reset_index(drop=True)


model, scaler, config, explainer = load_artifacts()
feature_cols = config["feature_cols"]
default_threshold = config["threshold"]

st.title("💳 Credit Card Fraud Detection — Live Scoring Demo")
st.caption(
    "Model: XGBoost trained on SMOTE-resampled data · "
    "Trained in `fraud_detection_project.ipynb` · "
    "Dataset is synthetic (mimics the Kaggle Credit Card Fraud dataset's structure & imbalance)."
)

sample_df = load_test_sample()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Pick a transaction")
    idx = st.selectbox(
        "Sample transaction (from held-out data)",
        options=sample_df.index,
        format_func=lambda i: (
            f"#{i}  |  Amount ${sample_df.loc[i, 'Amount']:.2f}  |  "
            f"{'🚨 actually fraud' if sample_df.loc[i, 'Class'] == 1 else '✅ actually legit'}"
        ),
    )
    row = sample_df.loc[[idx]]

    st.subheader("2. Or tweak it manually")
    amount = st.slider("Amount ($)", 0.0, 5000.0, float(row["Amount"].values[0]), step=1.0)
    hour = st.slider("Hour of day", 0, 23, int((row["Time"].values[0] // 3600) % 24))

    threshold = st.slider(
        "Decision threshold", 0.0, 1.0, float(default_threshold), step=0.01,
        help="Cost-optimal threshold from the notebook's Section 7 analysis is pre-filled. "
             "Lower = catch more fraud but more false alarms."
    )

# build the feature row used for scoring (keep PCA components from the picked sample,
# override Amount/Time with any manual tweaks)
score_row = row[feature_cols].copy()
score_row["Amount"] = amount
score_row["Time"] = hour * 3600

X_scaled = pd.DataFrame(scaler.transform(score_row), columns=feature_cols)
proba = model.predict_proba(X_scaled)[0, 1]
is_flagged = proba >= threshold

with col_right:
    st.subheader("3. Model verdict")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{proba*100:.2f}%")
    c2.metric("Threshold", f"{threshold*100:.0f}%")
    c3.metric("Verdict", "🚨 FLAGGED" if is_flagged else "✅ Approved")

    if row["Class"].values[0] == 1:
        st.caption("Ground truth for this sample: **actually fraud**")
    else:
        st.caption("Ground truth for this sample: **actually legit**")

    st.progress(min(float(proba), 1.0))

    st.subheader("4. Why? (SHAP explanation)")
    shap_values = explainer.shap_values(X_scaled)
    contrib = pd.Series(shap_values[0], index=feature_cols).sort_values(key=abs, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = np.where(contrib.sort_values() > 0, "#C44E52", "#4C72B0")
    contrib.sort_values().plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("SHAP value (pushes toward fraud →  |  ← pushes toward legit)")
    ax.set_title("Top features driving this prediction")
    st.pyplot(fig)

    st.caption(
        "Red bars push the score toward **fraud**, blue bars push toward **legit**. "
        "`V1`-`V28` are anonymized PCA components (as in the real Kaggle dataset); "
        "this mirrors the explanation an analyst would see in a production fraud-review queue."
    )

st.divider()
with st.expander("About this demo / the underlying model"):
    st.markdown(f"""
- **Model**: XGBoost classifier, trained on SMOTE-resampled data (see `fraud_detection_project.ipynb`)
- **Default threshold** ({default_threshold:.2f}): chosen to minimize expected cost
  (false negative ≈ $150, false positive ≈ $5) rather than a generic 0.5 cutoff
- **Data**: synthetic, built to mirror the real Kaggle *Credit Card Fraud Detection* dataset's
  structure (Time, Amount, 28 PCA features V1-V28) and severe class imbalance (~0.2% fraud)
- Swap in `data/creditcard.csv` from Kaggle and retrain (notebook Sections 1-9) to use real data
""")
