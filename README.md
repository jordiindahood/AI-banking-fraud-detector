# 🛡️ AI Banking Fraud Detector

A machine-learning project that detects fraudulent credit-card transactions using **XGBoost**, served through a **FastAPI** REST API and an interactive **Streamlit** dashboard.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Notebook Workflow](#notebook-workflow)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Launching the Application](#launching-the-application)
- [API Reference](#api-reference)

---

## Overview

Credit-card fraud is extremely rare — only **0.17 %** of transactions in the dataset are fraudulent — which makes it a challenging classification problem. This project walks through the complete ML pipeline:

1. **Exploratory Data Analysis** — understand the raw dataset
2. **Feature Engineering** — clean, scale, and prepare the data
3. **Model Selection** — compare models and data-balancing strategies
4. **Training & Tuning** — train XGBoost, tune hyperparameters, and optimise the decision threshold

The final model is served via a lightweight **FastAPI** endpoint and consumed by a **Streamlit** dashboard for real-time predictions.

---

## Dataset

**Kaggle — Credit Card Fraud Detection**
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

| Property | Value |
|---|---|
| Transactions | 284 807 |
| Fraud cases | 492 (0.17 %) |
| Features | 28 PCA components (V1–V28) + Amount + Time |
| Target | `Class` (0 = legitimate, 1 = fraud) |

> The features V1–V28 are the result of a PCA transformation applied to the original features for confidentiality reasons.

---

## Notebook Workflow

The project follows a structured, 4-notebook pipeline located in the `notebooks/` directory:

### Notebook 0 — Analysis of Original Dataset
**`0-Analysis_Of_Original_Dataset.ipynb`**

- Loads the raw CSV from `data/raw/creditcard.csv`
- Inspects shape, data types, missing values
- Visualises class distribution — highlights the extreme **99.83 % / 0.17 %** imbalance
- Analyses Amount and Time distributions
- **Conclusion**: The dataset is clean but highly imbalanced; a naive model would reach 99 % accuracy by predicting everything as legitimate

### Notebook 1 — EDA + Feature Engineering
**`1-EDA+FE.ipynb`**

- Removes duplicate rows
- Creates an **undersampled** subset (equal fraud / non-fraud) for balanced visualisation
- Computes correlation matrices on both the imbalanced and balanced data
- Identifies the most correlated PCA features (negative: V14, V12, V10 — positive: V11, V4, V2)
- **Scales** the `Amount` and `Time` columns using `RobustScaler` (V1–V28 are already PCA-scaled)
- Saves the processed dataset to `data/preprocessed/creditcard_Processed.csv`

### Notebook 2 — Model Selection
**`2-model_selection.ipynb`**

- Splits the processed data 80 / 20 (stratified)
- Tests **4 models** × **2 balancing strategies** using `StratifiedKFold` (5-fold) cross-validation:
  - Models: Logistic Regression, Linear SVM, XGBoost, Random Forest
  - Strategies: **SMOTE** (synthetic oversampling) vs **class-weight balancing**
- Compares precision, recall, F1-score, and PR-AUC
- Visualises confusion matrices for all 8 combinations
- **Conclusion: XGBoost + class-weight balancing** achieves the best trade-off

### Notebook 3 — Training & Tuning
**`3-train+tune.ipynb`**

- Splits the data 64 / 16 / 20 (train / validation / test, stratified)
- Computes `scale_pos_weight` from the training set
- Trains a **baseline XGBoost** and evaluates on validation
- Runs **RandomizedSearchCV** (30 iterations, 5-fold) to tune hyperparameters:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`
- Finds the **optimal decision threshold** (≈ 0.915) by maximising the F1-score on the precision-recall curve
- Evaluates on the held-out test set (confusion matrix, classification report, ROC-AUC, PR-AUC)
- Saves the final artifacts:
  - `models/xgb_fraud_model.pkl` — trained XGBoost model
  - `models/best_threshold.pkl` — optimal threshold

---

## Project Structure

```
AI-banking-fraud-detector/
├── data/
│   ├── raw/                        # Original Kaggle CSV
│   └── preprocessed/               # Scaled & cleaned CSV
├── models/
│   ├── xgb_fraud_model.pkl         # Trained XGBoost model
│   └── best_threshold.pkl          # Optimal decision threshold
├── notebooks/
│   ├── 0-Analysis_Of_Original_Dataset.ipynb
│   ├── 1-EDA+FE.ipynb
│   ├── 2-model_selection.ipynb
│   └── 3-train+tune.ipynb
├── notes/
│   ├── articles.txt                # Reference links
│   └── Definitions.txt             # ML concept definitions
├── src/
│   ├── __init__.py
│   ├── api.py                      # FastAPI prediction API
│   └── dashboard.py                # Streamlit dashboard
├── .env                            # Environment variables
├── config.yaml                     # Model & preprocessing config
├── requirements.txt                # Python dependencies
└── README.md                       # ← You are here
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- The dataset from Kaggle (download and place in `data/raw/creditcard.csv`)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/jordiindahood/AI-banking-fraud-detector.git
cd AI-banking-fraud-detector

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Reproduce the Model (optional)

If you want to retrain from scratch, run the notebooks in order:

```
notebooks/0-Analysis_Of_Original_Dataset.ipynb   →  EDA on raw data
notebooks/1-EDA+FE.ipynb                          →  Feature engineering
notebooks/2-model_selection.ipynb                 →  Model comparison
notebooks/3-train+tune.ipynb                      →  Final training + save
```

The saved model files in `models/` are already included for convenience.

---

## Launching the Application

You need **two terminals** — one for the API, one for the dashboard.

### 1. Start the API

```bash
# From the project root, with the venv active:
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

You should see:

```
INFO     Model loaded from …/models/xgb_fraud_model.pkl
INFO     Threshold loaded: 0.915011
INFO     Uvicorn running on http://127.0.0.1:8000
```

### 2. Start the Dashboard

```bash
# In a second terminal (venv active):
streamlit run src/dashboard.py
```

Open **http://localhost:8501** in your browser. Use the sidebar to enter feature values or click **Load Random Fraud / Legitimate** to auto-fill, then hit **Run Prediction**.

---

## API Reference

### `GET /health`

Health check.

```bash
curl http://127.0.0.1:8000/health
```

```json
{ "status": "ok" }
```

### `POST /predict`

Predict fraud probability for a single transaction.

**Request body** — 30 float features:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.35, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "scaled_amount": 0.24, "scaled_time": -0.99
  }'
```

**Response:**

```json
{
  "fraud_probability": 0.003142,
  "is_fraud": false,
  "threshold": 0.915011
}
```

| Field | Type | Description |
|---|---|---|
| `fraud_probability` | float | Probability the transaction is fraud (0–1) |
| `is_fraud` | bool | `true` if probability ≥ threshold |
| `threshold` | float | The decision threshold used (≈ 0.915) |
