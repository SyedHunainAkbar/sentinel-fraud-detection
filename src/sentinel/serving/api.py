"""FastAPI scoring endpoint for Sentinel."""
from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .. import config

app = FastAPI(title="Sentinel Fraud Scorer", version="0.1.0")

_MODEL = None
_FB = None


def _load():
    global _MODEL, _FB
    if _MODEL is None:
        _MODEL = joblib.load(config.MODELS_DIR / "xgboost.joblib")
        _FB = joblib.load(config.MODELS_DIR / "feature_builder.joblib")
    return _MODEL, _FB


class Transaction(BaseModel):
    trans_date_trans_time: str
    cc_num: int
    category: str
    amt: float
    gender: str
    state: str
    lat: float
    long: float
    city_pop: int
    dob: str
    unix_time: int
    merch_lat: float
    merch_long: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(txn: Transaction, threshold: float = 0.5):
    try:
        model, fb = _load()
        df = pd.DataFrame([txn.model_dump()])
        X, _, _ = fb.transform(df)
        prob = float(model.predict_proba(X)[:, 1][0])
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run `make train`.") from None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"Invalid transaction: {exc}") from exc
    return {"probability": prob, "decision": "alert" if prob >= threshold else "accept",
            "threshold": threshold}
