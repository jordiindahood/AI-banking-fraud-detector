import io

with open("src/dashboard.py", "r") as f:
    content = f.read()

# Let's completely rewrite dashboard.py as it is much easier to inject tabs correctly.
dashboard_code = """
\"\"\"
Streamlit dashboard for the Fraud Detection API.

Launch:  streamlit run src/dashboard.py
\"\"\"

import io
import streamlit as st
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "preprocessed" / "creditcard_Processed.csv"

FEATURE_NAMES = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "scaled_amount", "scaled_time",
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    \"\"\"
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* header bar */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.75;
        font-size: 0.95rem;
    }

    /* result cards */
    .result-card {
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }
    .result-card h2 {
        margin: 0 0 0.3rem 0;
        font-size: 2.4rem;
        font-weight: 700;
    }
    .result-card p {
        margin: 0;
        font-size: 0.95rem;
        opacity: 0.85;
    }
    .card-fraud {
        background: linear-gradient(135deg, #d32f2f, #b71c1c);
    }
    .card-safe {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
    }
    .card-prob {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
    }
    .card-threshold {
        background: linear-gradient(135deg, #6a1b9a, #4a148c);
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: #0e1117;
    }

    /* progress bar override */
    .stProgress > div > div > div > div {
        border-radius: 8px;
    }
    </style>
    \"\"\",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    \"\"\"
    <div class="main-header">
        <h1>🛡️ AI Fraud Detector</h1>
        <p>Real-time credit-card fraud prediction powered by XGBoost</p>
    </div>
    \"\"\",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — feature inputs
# ---------------------------------------------------------------------------
st.sidebar.title("📝 Transaction Features")

# Initialise session state for all features
if "features" not in st.session_state:
    st.session_state.features = {f: 0.0 for f in FEATURE_NAMES}


# --- Load sample row ---
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Fill")

if st.sidebar.button("🎲 Load Random Legitimate", use_container_width=True):
    if SAMPLE_CSV.exists():
        df = pd.read_csv(SAMPLE_CSV)
        sample = df[df["Class"] == 0].sample(1).iloc[0]
        for f in FEATURE_NAMES:
            st.session_state.features[f] = float(sample[f])
        st.sidebar.success("Loaded a random legitimate transaction")
    else:
        st.sidebar.error("Sample CSV not found")

if st.sidebar.button("🚨 Load Random Fraud", use_container_width=True):
    if SAMPLE_CSV.exists():
        df = pd.read_csv(SAMPLE_CSV)
        sample = df[df["Class"] == 1].sample(1).iloc[0]
        for f in FEATURE_NAMES:
            st.session_state.features[f] = float(sample[f])
        st.sidebar.success("Loaded a random fraud transaction")
    else:
        st.sidebar.error("Sample CSV not found")

st.sidebar.markdown("---")
st.sidebar.subheader("Feature Values")

# Number inputs for each feature
for feat in FEATURE_NAMES:
    st.session_state.features[feat] = st.sidebar.number_input(
        feat,
        value=st.session_state.features[feat],
        format="%.6f",
        key=f"input_{feat}",
    )


# ---------------------------------------------------------------------------
# Main area — prediction
# ---------------------------------------------------------------------------

tab_single, tab_batch = st.tabs(["Single Transaction", "Batch Prediction"])

with tab_single:
    col_btn, _ = st.columns([1, 3])

    with col_btn:
        predict_clicked = st.button(
            "🔍  Run Prediction",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
        # 1. Check API health
        try:
            health = requests.get(f"{API_URL}/health", timeout=5)
            health.raise_for_status()
        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ Cannot connect to the API.  "
                "Make sure it is running: `uvicorn src.api:app --host 127.0.0.1 --port 8000`"
            )
            st.stop()
        except requests.exceptions.RequestException as exc:
            st.error(f"API health check failed: {exc}")
            st.stop()

        # 2. Build payload
        payload = {feat: st.session_state.features[feat] for feat in FEATURE_NAMES}

        # 3. Call /predict
        with st.spinner("Analysing transaction…"):
            try:
                resp = requests.post(
                    f"{API_URL}/predict", json=payload, timeout=10
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")
                st.stop()

        # 4. Display results
        prob = result["fraud_probability"]
        is_fraud = result["is_fraud"]
        thresh = result["threshold"]

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        with c1:
            card_class = "card-fraud" if is_fraud else "card-safe"
            verdict = "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE"
            st.markdown(
                f'<div class="result-card {card_class}">'
                f"<h2>{verdict}</h2>"
                f"<p>Model verdict</p></div>",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f'<div class="result-card card-prob">'
                f"<h2>{prob:.4%}</h2>"
                f"<p>Fraud probability</p></div>",
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f'<div class="result-card card-threshold">'
                f"<h2>{thresh:.4f}</h2>"
                f"<p>Decision threshold</p></div>",
                unsafe_allow_html=True,
            )

        # Probability bar
        st.markdown("")
        st.markdown("#### Probability Gauge")
        st.progress(min(prob, 1.0))

        # Feature summary table
        st.markdown("#### Feature Values Sent")
        feat_df = pd.DataFrame(
            list(payload.items()), columns=["Feature", "Value"]
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    else:
        # Landing state
        st.info(
            "👈 Fill in the transaction features in the sidebar (or click **Load Random**), "
            "then press **Run Prediction**."
        )
        st.markdown("---")

        # Quick stats section
        st.markdown("#### ℹ️ About This Tool")
        st.markdown(
            \"\"\"
            This dashboard connects to the **FastAPI prediction API** which runs an
            **XGBoost** classifier trained on the
            [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
            dataset.

            - **30 input features** — 28 PCA components (V1–V28) plus `scaled_amount`
              and `scaled_time`
            - **Custom threshold** (≈ 0.915) optimised for the best F1-score on the
              validation set
            - Transactions with a fraud probability **≥ threshold** are flagged as
              fraudulent
            \"\"\"
        )

with tab_batch:
    st.markdown("### Paste Transactions (CSV format)")
    st.markdown("Paste comma-separated values. Make sure the columns match the required 30 features.")
    
    csv_text = st.text_area("CSV Input", height=200, placeholder="V1,V2,V3...,scaled_amount,scaled_time\n-1.35,-0.07,2.54...,0.24,-0.99")
    
    if st.button("🔍 Predict Batch", type="primary", use_container_width=True):
        if not csv_text.strip():
            st.error("Please paste some CSV data first.")
        else:
            try:
                # Attempt to parse
                df_batch = pd.read_csv(io.StringIO(csv_text))
                
                # Check for missing features
                missing_feats = [f for f in FEATURE_NAMES if f not in df_batch.columns]
                
                # If no header provided, and column count matches, assume order
                if len(df_batch.columns) == len(FEATURE_NAMES) and len(missing_feats) == len(FEATURE_NAMES):
                    df_batch.columns = FEATURE_NAMES
                    missing_feats = []
                    
                if missing_feats:
                    st.error(f"Missing required columns: {', '.join(missing_feats)}")
                else:
                    # Convert to list of dicts
                    payload = df_batch[FEATURE_NAMES].to_dict('records')
                    
                    with st.spinner("Analysing batch…"):
                        resp = requests.post(
                            f"{API_URL}/predict_batch", json=payload, timeout=30
                        )
                        resp.raise_for_status()
                        results = resp.json()
                        
                    # Add results to dataframe
                    df_batch["fraud_probability"] = [r["fraud_probability"] for r in results]
                    df_batch["is_fraud"] = [r["is_fraud"] for r in results]
                    
                    st.success(f"Successfully analysed {len(df_batch)} transactions.")
                    
                    # Highlight fraud rows
                    def color_fraud(val):
                        color = '#ff4b4b' if val else ''
                        return f'background-color: {color}'
                    
                    st.dataframe(
                        df_batch.style.applymap(color_fraud, subset=['is_fraud']),
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Error processing batch: {e}")

"""

with open("src/dashboard.py", "w") as f:
    f.write(dashboard_code)

