import re

with open("src/api.py", "r") as f:
    content = f.read()

batch_endpoint = """
@app.post("/predict_batch", response_model=list[PredictionResponse])
async def predict_batch(transactions: list[TransactionInput]):
    \"\"\"Return fraud predictions for a batch of transactions.\"\"\"
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
"""

content = content + "\n" + batch_endpoint

with open("src/api.py", "w") as f:
    f.write(content)
