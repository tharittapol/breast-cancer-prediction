"""
Inference utilities for the breast-cancer prediction service.

- Lazily load a trained sklearn model from disk (model/model.joblib).
- Accept input either as:
    * JSON (list[float] or list[list[float]])
    * CSV file bytes
- Use shared preprocessing helpers to clean/validate inputs.
- Run predictions and (if available) class probabilities.
- Log basic prediction metadata (batch size + latency) to predictions.log.
"""

import time
import joblib
from pathlib import Path

from .pre_process import from_json_payload, from_csv_bytes
from .logger import get_pred_logger

# Path to the serialized sklearn model (saved with joblib.dump)
MODEL_PATH = Path("model/model.joblib")

# Dedicated logger for prediction events (see logger.py)
_pred_logger = get_pred_logger()

# Global cached model instance; loaded on first use
_model = None


def load_model():
    """
    Load and cache the ML model from disk.

    - Uses a global variable `_model` to avoid reloading on every request.
    - Returns the loaded model instance.
    """
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_from_json(payload: list) -> dict:
    """
    Run prediction given a JSON payload (list[float] or list[list[float]]).

    Steps:
        1. Load the model (lazy).
        2. Convert JSON into a cleaned 2D ndarray via `from_json_payload`.
        3. Measure prediction time (latency in ms).
        4. Return predicted labels, probabilities, and latency.

    Returns:
        dict with keys:
            - "pred": list of predicted class labels
            - "proba": list of class probabilities or None
            - "latency_ms": float, inference time in milliseconds
    """
    model = load_model()
    X = from_json_payload(payload)

    t0 = time.time()
    y = model.predict(X).tolist()  # Convert numpy array -> JSON-friendly list

    # RandomForest models expose predict_proba
    probs = getattr(model, "predict_proba", None)
    p = probs(X).tolist() if probs else None

    latency_ms = round((time.time() - t0) * 1000, 3)

    _pred_logger.info({"n": len(X), "latency_ms": latency_ms})

    return {"pred": y, "proba": p, "latency_ms": latency_ms}


def predict_from_csv(file_bytes: bytes) -> dict:
    """
    Run prediction given raw CSV bytes (uploaded file).

    Steps:
        1. Load the model (lazy).
        2. Parse CSV bytes into a DataFrame, then ndarray via `from_csv_bytes`.
        3. Measure prediction time (latency in ms).
        4. Return predicted labels, probabilities, and latency.

    Returns:
        dict with keys:
            - "pred": list of predicted class labels
            - "proba": list of class probabilities or None
            - "latency_ms": float, inference time in milliseconds
    """
    model = load_model()
    X = from_csv_bytes(file_bytes)

    t0 = time.time()
    y = model.predict(X).tolist()

    probs = getattr(model, "predict_proba", None)
    p = probs(X).tolist() if probs else None

    latency_ms = round((time.time() - t0) * 1000, 3)

    _pred_logger.info({"n": len(X), "latency_ms": latency_ms})

    return {"pred": y, "proba": p, "latency_ms": latency_ms}
