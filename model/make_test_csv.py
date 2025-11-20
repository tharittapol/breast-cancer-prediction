from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

def main():
    data = load_breast_cancer(as_frame=True)

    X = data.frame.drop(columns=["target"])

    X_test = X.head(10)

    out_path = Path("data") / "samples" / "test_samples.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    X_test.to_csv(out_path, index=False)
    print("Saved test CSV to", out_path.resolve())

if __name__ == "__main__":
    main()
