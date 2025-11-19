"""
Generate a simple performance report from prediction logs.

This script:
- Reads `logs/predictions.log` written by the prediction service.
- Extracts JSON objects containing:
    * "n"          : batch size per request
    * "latency_ms" : inference latency in milliseconds
- Aggregates them into a pandas DataFrame.
- Computes summary statistics and saves them as `reports/artifacts/summary.csv`.

Intended usage:
    python reports/make_report.py
"""

import json
import pandas as pd
from pathlib import Path

# Path to the prediction log file produced by get_pred_logger()
LOG = Path("logs/predictions.log")

# Directory where report artifacts (like summary.csv) will be stored
OUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_lines() -> pd.DataFrame:
    """
    Parse the prediction log into a DataFrame.

    - Each log line is expected to contain JSON after "INFO predict: ".
    - Extract that JSON, decode it, and collect rows with fields "n" and "latency_ms".
    - If the log file does not exist, return an empty DataFrame with the right columns.
    """
    rows = []

    # If no log file yet, return empty DataFrame with expected columns
    if not LOG.exists():
        return pd.DataFrame(columns=["n", "latency_ms"])

    # Read the log file line by line
    for line in LOG.read_text().splitlines():
        try:
            # Take the part after "INFO predict: " which should be a JSON object
            payload_text = line.split("INFO predict: ")[-1]
            obj = json.loads(payload_text)
            rows.append(obj)
        except Exception:
            # Ignore any lines that do not match the expected pattern/JSON
            pass

    # Convert accumulated rows into a DataFrame
    return pd.DataFrame(rows)


def main() -> None:
    """
    Entry point for generating the report.

    - Parse logs into a DataFrame.
    - If there are no records, print a message and exit.
    - Otherwise, compute:
        * total_requests
        * avg_batch_size
        * avg_latency_ms
        * p95_latency_ms
      and save them to summary.csv.
    """
    df = parse_lines()

    if df.empty:
        print("No prediction logs yet.")
        return

    # Compute summary statistics from the log data
    summary = {
        "total_requests": int(len(df)),
        "avg_batch_size": float(df["n"].mean()),
        "avg_latency_ms": float(df["latency_ms"].mean()),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
    }

    # Save a one-row CSV with the metrics
    out_path = OUT_DIR / "summary.csv"
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    print("Saved report to", out_path)


if __name__ == "__main__":
    main()
