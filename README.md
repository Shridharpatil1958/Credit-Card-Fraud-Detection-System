# 💳 Credit Card Fraud Detection using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6F00?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### 🚀 End-to-End Machine Learning Project for Detecting Credit Card Fraud

*Handling highly imbalanced data using advanced resampling techniques, cost-sensitive learning, explainable AI, and an interactive Streamlit application.*

---

</div>

# 📌 Project Overview

Credit card fraud detection is one of the most challenging classification problems because fraudulent transactions account for only **0.2%** of all transactions.

This project demonstrates an **industry-style fraud detection pipeline** covering:

* 📊 Exploratory Data Analysis (EDA)
* ⚖️ Imbalanced Data Handling
* 🤖 Multiple Machine Learning Models
* 📈 Model Performance Comparison
* 🎯 Cost-Based Threshold Optimization
* 🔍 SHAP Explainability
* 🌐 Interactive Streamlit Dashboard
* 💾 Model Serialization & Deployment

Instead of maximizing **accuracy**, this project focuses on metrics that actually matter in fraud detection such as:

* Precision-Recall AUC
* ROC-AUC
* F1 Score
* Cost Optimization
* Business Impact

---

# ✨ Features

✅ Synthetic Credit Card Dataset Generator

✅ Extensive Exploratory Data Analysis

✅ Handles Severe Class Imbalance

✅ Multiple Resampling Strategies

* Random Under Sampling
* SMOTE
* ADASYN
* SMOTE + Tomek Links
* Class Weighting

✅ Multiple Machine Learning Models

* Logistic Regression
* XGBoost
* Isolation Forest

✅ Cost-Based Threshold Optimization

✅ SHAP Explainability

✅ Model Comparison Dashboard

✅ Interactive Streamlit Application

---

# 📂 Project Structure

```text
fraud_detection/
│
├── README.md
│
├── data/
│   ├── generate_data.py
│   └── creditcard.csv
│
├── fraud_detection_project.ipynb
│
├── models/
│   ├── fraud_xgb_model.pkl
│   ├── scaler.pkl
│   └── model_config.pkl
│
├── outputs/
│   ├── eda_plots.png
│   ├── model_comparison.png
│   ├── shap_summary.png
│   └── threshold_curve.png
│
└── app/
    └── streamlit_app.py
```

---

# 📊 Dataset

Since GitHub cannot distribute the original Kaggle dataset, this repository includes a **synthetic dataset generator** that closely mimics the real credit card fraud dataset.

### Dataset Characteristics

| Feature      | Value                |
| ------------ | -------------------- |
| Transactions | 60,000               |
| Fraud Rate   | ~0.2%                |
| Features     | Time, V1–V28, Amount |
| Target       | Class                |

### Synthetic Data Includes

* Realistic fraud ratio
* PCA-style anonymous features
* Late-night fraud behavior
* Card testing transactions
* High-value fraud
* Label noise
* Overlapping distributions

---

## 📥 Using the Real Dataset

Download the original dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Replace

```
data/creditcard.csv
```

with the Kaggle version and rerun the notebook.

No code changes are required because the schema is identical.

---

# ⚙️ Machine Learning Pipeline

```text
Raw Data
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Train/Test Split
    │
    ▼
Feature Scaling
    │
    ▼
Imbalanced Data Handling
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Threshold Optimization
    │
    ▼
SHAP Explainability
    │
    ▼
Model Deployment
```

---

# 🧠 Models Used

| Model               | Purpose                        |
| ------------------- | ------------------------------ |
| Logistic Regression | Baseline Linear Classifier     |
| XGBoost             | Gradient Boosting              |
| Isolation Forest    | Unsupervised Anomaly Detection |

---

# ⚖️ Imbalance Handling Techniques

This project compares several approaches for learning from highly imbalanced datasets.

* Random Undersampling
* SMOTE
* ADASYN
* SMOTE + Tomek
* Class Weighting
* XGBoost `scale_pos_weight`

---

# 📈 Model Evaluation Metrics

Instead of Accuracy, this project evaluates models using:

* ✅ Precision
* ✅ Recall
* ✅ F1 Score
* ✅ ROC-AUC
* ✅ Precision-Recall AUC
* ✅ Confusion Matrix
* ✅ Cost-Based Threshold

---

# 💰 Cost-Based Threshold Optimization

Traditional classification uses:

```python
threshold = 0.50
```

This project searches for the **optimal business threshold** by minimizing the cost of:

* False Positives
* False Negatives

This creates a fraud detection system aligned with real-world financial risk.

---

# 🔍 Explainable AI (SHAP)

The notebook includes:

* Global Feature Importance
* SHAP Summary Plot
* Individual Transaction Explanation

This allows fraud analysts to understand **why** a transaction was flagged.

---

# 🌐 Streamlit Application

The repository includes a complete Streamlit interface where users can:

* Enter transaction details
* Predict fraud probability
* View prediction confidence
* Display fraud risk
* Load saved model automatically

Run the application using:

```bash
streamlit run app/streamlit_app.py
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/fraud_detection.git

cd fraud_detection
```

Install dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap streamlit matplotlib seaborn joblib
```

(Optional) Generate synthetic data

```bash
python data/generate_data.py
```

Run the notebook

```bash
jupyter notebook fraud_detection_project.ipynb
```

Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

# 📷 Project Screenshots

> Add screenshots here after running the notebook.

```
outputs/
├── class_distribution.png
├── fraud_amount_distribution.png
├── correlation_heatmap.png
├── model_comparison.png
├── shap_summary.png
├── threshold_curve.png
└── streamlit_demo.png
```

---

# 📦 Saved Models

The trained models are stored in:

```text
models/
├── fraud_xgb_model.pkl
├── scaler.pkl
└── model_config.pkl
```

These files are automatically loaded by the Streamlit application.

---

# 📚 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

# 🎯 Key Learning Outcomes

* Working with highly imbalanced datasets
* Comparing multiple resampling techniques
* Fraud detection using machine learning
* Cost-sensitive classification
* Explainable AI using SHAP
* Building production-ready ML pipelines
* Model deployment with Streamlit

---

# ⭐ Key Takeaway

> **Accuracy is misleading in fraud detection.**

With only **0.2% fraudulent transactions**, a model predicting **"No Fraud"** for every transaction achieves nearly **99.8% accuracy** while providing zero practical value.

This project emphasizes **Precision-Recall AUC**, **business-aware threshold optimization**, and **explainable predictions** to build a fraud detection system suitable for real-world applications.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

**Built with ❤️ using Python, Machine Learning, XGBoost, SHAP & Streamlit**

</div>
