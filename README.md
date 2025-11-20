# 🩺 Breast Cancer Prediction API (FastAPI + scikit-learn)

Production-style **ML inference service** for the **Breast Cancer Wisconsin (Diagnostic)** dataset, built with:

- 🧠 `scikit-learn` (Breast Cancer Wisconsin dataset + RandomForest)
- ⚙️ `FastAPI` for a clean, typed HTTP API
- 📦 `joblib` for model persistence
- 🪵 rotating file logs + simple reporting script
- 🐳 optional Docker image for deployment

The project demonstrates a full ML workflow:

> **Train pipeline model → dump model → serve prediction via FastAPI → log predictions/apps → generate report**

---

## 📁 Project Structure

```text
breast-cancer-prediction/
├─ app/
│  ├─ main.py            # FastAPI app (health + /predict)
│  ├─ inference.py       # Model loading + prediction helpers
│  ├─ pre_process.py     # Input cleaning for JSON / CSV
│  ├─ logger.py          # Rotating file loggers (app + predictions)
│  ├─ schemas.py         # Pydantic schema for JSON payload
│
├─ model/
│  ├─ train_pipeline.py  # Train pipeline, select features, save model + metrics
│  ├─ make_test_csv.py   # Generate a small CSV with sample inputs
│  ├─ model.joblib       # Trained sklearn Pipeline (created after train)
│  ├─ metrics.json       # Accuracy / F1 metrics for the saved model
│
├─ reports/
│  ├─ make_report.py     # Aggregate logs into a CSV summary
│
├─ logs/
│  ├─ app.log            # App-level logs
│  └─ predictions.log    # Inference logs (batch size + latency)
│
├─ requirements.txt
├─ Dockerfile
├─ Makefile              # Linux/macOS helpers (train, build, run, report)
└─ README.md
```
---
## 🧪 Model Overview
- **Dataset**: `Breast Cancer Wisconsin (Diagnostic) from sklearn.datasetsload_breast_cancer`

- **Features**: 30 numeric features (mean / error / worst metrics)

- **Target**: binary (1: malignant / 0: benign)

- **Base model**: RandomForestClassifier

- **Feature selection** (done in `model/train_pipeline.py`)
- **Artifacts saved:**
  - `model/model.joblib` – the final `Pipeline`
  - `model/metrics.json` – `{"accuracy": ..., "f1_score": ...}`

- The pipeline is fit on raw arrays (`X_train.values`) so it is compatible with the FastAPI preprocessing, which returns `ndarray`.

At inference time, the service expects 30 features (in the original order). The pipeline then internally selects only `keep_idx` columns.

---
## ⚙️ Setup (Local)
**1. Clone repo**

```powershell
git clone https://github.com/tharittapol/breast-cancer-prediction.git
cd breast-cancer-prediction
```

**2. create virtualenv**
```powershell
# Create & activate venv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Or on macOS/Linux:
# python -m venv .venv
# source .venv/bin/activate
```
**Install dependencies**
```powershell
pip install -r requirements.txt
```

**Train the model (Optional)**
```powershell
python -m model.train_pipeline
```
After running, you should see:

- `model/model.joblib`

- `model/metrics.json`

---

## 🚀 Run the FastAPI Server
With the virtualenv active:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Endpoints:

- Health check: GET `http://localhost:8000/health`

- API docs (Swagger UI): `http://localhost:8000/docs`

---
## 📡 API Usage
### Endpoint

`POST /predict`

Supports:

- `Content-Type: application/json`

  - `{"data": [float, ..., float]}` → single row

  - `{"data": [[...], [...], ...]}` → multiple rows

- `Content-Type: multipart/form-data`

  - CSV file under field name `file`

### Response format
The API returns one object per input row:

```json
{
  "results": [
    { "index": 0, "pred": 1, "proba": [0.12, 0.88] },
    { "index": 1, "pred": 0, "proba": [0.90, 0.10] }
  ],
  "latency_ms": 0.42
}
```
- `index` – index of the row within this request batch.

- `pred` – predicted class label (0 or 1).

- `proba` – class probabilities `[p(class=0), p(class=1)]` if `predict_proba` is available.

- `latency_ms` – total inference time in milliseconds.

---
## 🧩 Example Requests (PowerShell)
### 1) JSON – single row (30 features)
```powershell
$body = @{
    data = @(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
             1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,
             2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0)
} | ConvertTo-Json -Depth 3

$response = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST `
  -ContentType "application/json" -Body $body

$response.results
$response.latency_ms
```
### 2) JSON – multiple rows
```powershell
$body = @{
    data = @(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
             1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,
             2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0),

	   @(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,
             1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.0,
             2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0)
} | ConvertTo-Json -Depth 3

$response = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST `
  -ContentType "application/json" -Body $body

$response.results
$response.latency_ms
```
PowerShell will show output row by row, e.g.:
```text
index pred proba
----- ---- -----
    0    1 {0.12, 0.88}
    1    0 {0.90, 0.10}
```

---
## 📄 CSV Testing
### Generate sample CSV from the real dataset (optional)
Use the helper script:
```powershell
python -m model.make_test_csv
```
This will save them as: `data/samples/test_samples.csv`

### Send CSV with curl (Windows or Linux)
```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -F "file=@data/samples/test_samples.csv;type=text/csv"
```

---
## 🧾 Logging & Reporting
- `logs/app.log` – application-level logging.

- `logs/predictions.log` – prediction logs:
  - Each line contains a JSON object with:

    - `n` – batch size (rows per request).

    - `latency_ms` – inference latency in ms.  

### Generate a latency report
```powershell
python -m reports.make_report
```
This will save them as: `reports/artifacts/summary.csv`  

Example:
```csv
total_requests,avg_batch_size,avg_latency_ms,p95_latency_ms
42,3.12,1.45,3.20
```

---
## ⚙️ Setup Docker🐳 (Optional)
If you have Docker installed and want to run the service in a container:
```powershell
docker build -t breast-cancer-prediction:latest .
```
Run the container:
```powershell
docker run --rm -d -p 8000:8000 \
  --name breast-cancer-prediction \
  -v "$PWD/logs:/app/logs" \
  breast-cancer-prediction:latest
```
Then open:

- `http://localhost:8000/health`

- `http://localhost:8000/docs`

---
## 🧰 Makefile Targets (Linux/macOS)
If you have `make` available:
```bash
# Train model (model/train_pipeline.py)
make train

# Build Docker image
make build

# Run container
make run

# Stop container
make stop

# Tail container logs
make logs

# Run tests (if you add tests/)
make test

# Generate prediction latency report
make report
```

On Windows (PowerShell), you can call the underlying commands directly, e.g.:
```powershell
# Train (Optional)
(.venv) python -m model.train_pipeline

# Build image
docker build -t breast-cancer-prediction:latest .

# Run container
docker run --rm -d -p 8000:8000 `
  --name breast-cancer-prediction `
  -v "${PWD}\logs:/app/logs" `
  breast-cancer-prediction:latest
```

---

## ✅ Status & Future Ideas

Current features:

- ✅ End-to-end training pipeline with feature selection

- ✅ Saved model + metrics + feature mapping

- ✅ JSON & CSV prediction API

- ✅ Logging + simple latency report

- ✅ Example clients (PowerShell + curl)

Possible future extensions:

- 🔐 API keys / auth

- 📈 Metrics export to Prometheus / Grafana

- 🖥️ Web UI for manual predictions and visual explanations

- 📊 SHAP or other explainability tooling

---
If you find this project useful or educational, feel free to ⭐ the repo or fork it and plug in your own models 🚀
```makefile
::contentReference[oaicite:0]{index=0}
```