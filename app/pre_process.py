import numpy as np
import pandas as pd
from typing import List

N_FEATURES = 23

def clean_array(arr: np.ndarray) -> np.ndarray:
    """
    Ensure the input is a 2D array with N_FEATURES columns,
    coerce values to numeric, impute missing values with column medians,
    and return a float ndarray.
    """
    # Ensure 2D shape: (n_samples, n_features)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {arr.shape[1]}")
    df = pd.DataFrame(arr)
    # Convert to numeric, turning invalid entries into NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    # Impute NaN with column medians
    df = df.fillna(df.median(numeric_only=True))
    return df.values.astype(float)

def from_json_payload(payload) -> np.ndarray:
    """
    Convert a JSON payload (list of floats or list of list of floats)
    into a cleaned 2D numpy array ready for prediction.
    """
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("JSON must be a non-empty list")
    else:
        arr = np.array(payload, dtype=float)
    return clean_array(arr)

def from_csv_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Read a CSV file from raw bytes, convert to ndarray, and clean it.
    The CSV is expected to contain exactly N_FEATURES columns.
    """
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    return clean_array(df.values)
