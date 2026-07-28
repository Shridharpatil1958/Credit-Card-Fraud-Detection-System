"""
Generate a synthetic credit-card-transaction dataset that mimics the
structure and statistical properties of the well-known Kaggle
'Credit Card Fraud Detection' dataset (284,807 txns, 0.17% fraud,
28 PCA features V1-V28 + Time + Amount + Class).

Why synthetic? This sandbox can't reach kaggle.com. Everything below
(EDA, SMOTE, models, evaluation, app) works identically on the real
Kaggle CSV -- just swap the file in step 1 of the notebook.
To use the real data yourself: download from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and drop creditcard.csv into the data/ folder.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

RANDOM_STATE = 42
N_SAMPLES = 60_000
FRAUD_RATIO = 0.0017  # matches real-world ~0.17%


def generate_dataset():
    rng = np.random.default_rng(RANDOM_STATE)

    # 28 latent "PCA-like" features (V1-V28), informative + noise mix,
    # with a separable-but-overlapping fraud cluster (realistic difficulty)
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=28,
        n_informative=12,
        n_redundant=6,
        n_clusters_per_class=3,
        weights=[1 - FRAUD_RATIO, FRAUD_RATIO],
        flip_y=0.001,          # small label noise, like real data
        class_sep=0.9,
        random_state=RANDOM_STATE,
    )

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Class"] = y

    # make_classification's weights are approximate -> recompute actual counts
    fraud_mask = df["Class"] == 1
    n_fraud = int(fraud_mask.sum())
    n_legit = N_SAMPLES - n_fraud

    # Time: seconds elapsed over a 2-day window, fraud slightly more
    # likely at odd hours (a realistic, learnable pattern)
    hours = rng.uniform(0, 48, size=N_SAMPLES)
    hours[fraud_mask.values] = np.where(
        rng.random(fraud_mask.sum()) < 0.4,
        rng.uniform(0, 5, size=fraud_mask.sum()),   # late-night skew
        hours[fraud_mask.values],
    )
    df["Time"] = np.sort(hours * 3600).round(0)  # not truly sorted per-row link, illustrative only
    df["Time"] = (hours * 3600).round(0)

    # Amount: legit txns log-normal small purchases; fraud txns
    # bimodal (many tiny "card testing" charges + a few large ones)
    legit_amt = rng.lognormal(mean=3.0, sigma=1.1, size=n_legit)
    small_fraud = rng.lognormal(mean=1.0, sigma=0.6, size=int(n_fraud * 0.6))
    large_fraud = rng.lognormal(mean=5.5, sigma=0.8, size=n_fraud - len(small_fraud))
    fraud_amt = np.concatenate([small_fraud, large_fraud])
    rng.shuffle(fraud_amt)

    amounts = np.empty(N_SAMPLES)
    amounts[~fraud_mask.values] = legit_amt
    amounts[fraud_mask.values] = fraud_amt
    df["Amount"] = amounts.round(2)

    # reorder columns to match the familiar Kaggle layout
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df = df[cols].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "/home/claude/fraud_detection/data/creditcard.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows -> {out_path}")
    print(df["Class"].value_counts(normalize=True))
