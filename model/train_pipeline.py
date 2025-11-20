import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


def main():
    # -------------------------------------------------------------------------
    # 1) Load data as DataFrame (X = 30 features, y = target)
    # -------------------------------------------------------------------------
    data = load_breast_cancer(as_frame=True)
    df = data.frame  # includes all features + 'target' column

    X = df.drop(columns=["target"])
    y = df["target"]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------------------------------------------------
    # 2) Baseline RF for permutation importance
    # -------------------------------------------------------------------------
    rf_base = RandomForestClassifier(
        random_state=42
    )
    rf_base.fit(X_train, y_train)

    r = permutation_importance(
        rf_base,
        X_train,
        y_train,
        random_state=42,
    )

    perm = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "pi_mean": r.importances_mean,
            }
        )
        .sort_values("pi_mean", ascending=False)
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # 3) Choose features to keep
    # -------------------------------------------------------------------------
    keep_names = perm.loc[perm.pi_mean > 0, "feature"].tolist() or perm.head(12)[
        "feature"
    ].tolist()
    print(f"Keeping {len(keep_names)} features:")
    print(keep_names)

    # Map feature names -> column indices (0..29),
    # so the pipeline can work with plain ndarray at inference time
    keep_idx = [feature_names.index(name) for name in keep_names]

    # -------------------------------------------------------------------------
    # 4) Build Pipeline: selector (by index) -> RandomForest
    # -------------------------------------------------------------------------
    selector = ColumnTransformer(
        transformers=[
            ("keep", "passthrough", keep_idx),
        ],
        remainder="drop",
    )

    rf_selected = RandomForestClassifier(
        random_state=42
    )

    pipe = Pipeline(
        steps=[
            ("select", selector),
            ("rf", rf_selected),
        ]
    )

    # Fit pipeline on raw array
    pipe.fit(X_train.values, y_train)

    # -------------------------------------------------------------------------
    # 5) Evaluate on test set
    # -------------------------------------------------------------------------
    y_pred = pipe.predict(X_test.values)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy (rf_selected pipeline): {acc:.4f}")
    print(f"F1-score (rf_selected pipeline): {f1:.4f}")

    # -------------------------------------------------------------------------
    # 6) Save model, metrics
    # -------------------------------------------------------------------------
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    out_path = model_dir / "model.joblib"
    joblib.dump(pipe, out_path)
    print("Saved model to", out_path.resolve())

    metrics_path = model_dir / "metrics.json"
    metrics = {
        "accuracy": float(acc),
        "f1_score": float(f1),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", metrics_path.resolve())


if __name__ == "__main__":
    main()
