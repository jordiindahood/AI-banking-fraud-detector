"""
FastAPI prediction API for the XGBoost fraud detection model.

Endpoints:
    GET  /health   — Health check
    POST /predict  — Predict fraud probability for a transaction
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — resolve relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_fraud_model.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "best_threshold.pkl"

# ---------------------------------------------------------------------------
# Global holders (populated at startup)
# ---------------------------------------------------------------------------
model = None
threshold: float = 0.5  # safe default, overridden on load


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, threshold

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))

    logger.info("Model loaded from %s", MODEL_PATH)
    logger.info("Threshold loaded: %.6f", threshold)
    yield  # app is running
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Predict whether a credit-card transaction is fraudulent.",
    version="1.0.0",
    lifespan=lifespan,
)

# TODO(security): Restrict origins to the actual frontend domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    """The 30 features expected by the XGBoost model."""

    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    scaled_amount: float = Field(..., description="Robust-scaled transaction amount")
    scaled_time: float = Field(..., description="Robust-scaled transaction time")


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(
        ..., description="Probability that the transaction is fraud (0–1)"
    )
    is_fraud: bool = Field(
        ..., description="True if probability ≥ threshold"
    )
    threshold: float = Field(
        ..., description="Decision threshold used"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionInput):
    """Return the fraud prediction for a single transaction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    features = np.array(
        [[
            transaction.V1, transaction.V2, transaction.V3, transaction.V4,
            transaction.V5, transaction.V6, transaction.V7, transaction.V8,
            transaction.V9, transaction.V10, transaction.V11, transaction.V12,
            transaction.V13, transaction.V14, transaction.V15, transaction.V16,
            transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27, transaction.V28,
            transaction.scaled_amount, transaction.scaled_time,
        ]]
    )

    proba = float(model.predict_proba(features)[0, 1])

    return PredictionResponse(
        fraud_probability=round(proba, 6),
        is_fraud=proba >= threshold,
        threshold=round(threshold, 6),
    )


@app.post("/predict_batch", response_model=list[PredictionResponse])
async def predict_batch(transactions: list[TransactionInput]):
    """Return fraud predictions for a batch of transactions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not transactions:
        return []

    features = np.array([
        [
            t.V1, t.V2, t.V3, t.V4, t.V5, t.V6, t.V7, t.V8,
            t.V9, t.V10, t.V11, t.V12, t.V13, t.V14, t.V15, t.V16,
            t.V17, t.V18, t.V19, t.V20, t.V21, t.V22, t.V23, t.V24,
            t.V25, t.V26, t.V27, t.V28, t.scaled_amount, t.scaled_time
        ] for t in transactions
    ])

    probas = model.predict_proba(features)[:, 1]

    return [
        PredictionResponse(
            fraud_probability=round(float(p), 6),
            is_fraud=bool(p >= threshold),
            threshold=round(float(threshold), 6)
        )
        for p in probas
    ]
