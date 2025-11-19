"""
FastAPI entrypoint for the Breast Cancer Wisconsin prediction service.

Features:
- GET /health       : simple health check endpoint.
- POST /predict     : accepts either
    * JSON          : {"data": [...]} or {"data": [[...], [...]]}
    * CSV upload    : multipart/form-data with `file` field.
- Routes incoming data to the appropriate inference function.
- Handles validation errors (400), unsupported content types (415),
  and unexpected internal errors (500) with logging.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import ValidationError
from .schemas import PredictJSON
from .inference import predict_from_json, predict_from_csv
from .logger import get_app_logger

# Create FastAPI application with a descriptive title
app = FastAPI(title="BC-Wisconsin FastAPI")

# Application-level logger (see logger.py)
alog = get_app_logger()


@app.get("/health")
def health() -> dict:
    """
    Health check endpoint.

    Returns a simple JSON object indicating that the service is running.
    Used by uptime probes / monitoring.
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile | None = File(default=None),   # Uploaded CSV file (if multipart/form-data)
) -> dict:
    """
    Main prediction endpoint.

    Behavior:
        - If Content-Type is application/json:
            * Expect a PredictJSON body.
            * Validate and pass payload.data to `predict_from_json`.
        - If Content-Type is multipart/form-data:
            * Expect a CSV file in the `file` field.
            * Read bytes and pass to `predict_from_csv`.
        - Otherwise:
            * Return 415 Unsupported Media Type.

    Errors:
        - Raises 400 on validation errors (e.g., wrong shape, empty body).
        - Logs and raises 500 on unexpected internal errors.
    """
    # Determine how the client sent the data (JSON vs multipart)
    ctype = request.headers.get("content-type", "")

    try:
        if "application/json" in ctype:
            # Read raw JSON body
            body = await request.json()

            # Validate & parse with Pydantic
            try:
                payload = PredictJSON(**body)
            except ValidationError as e:
                raise HTTPException(400, f"Invalid JSON body: {e}") from e

            return predict_from_json(payload.data)

        elif "multipart/form-data" in ctype:
            # File upload mode: CSV file must be provided under field name 'file'
            if file is None:
                raise HTTPException(400, "CSV file required under field 'file'")

            content = await file.read()
            return predict_from_csv(content)

        else:
            # Any other content type is not supported for this endpoint
            raise HTTPException(
                415,
                "Unsupported Content-Type. Use JSON or multipart/form-data",
            )
        
    except HTTPException:
        # Let FastAPI return the correct status code (400/415, etc.)
        raise

    except ValueError as e:
        # Typically raised by preprocessing/validation (e.g., wrong number of features)
        raise HTTPException(400, f"Validation error: {e}")

    except Exception as e:
        # Log full stack trace for debugging, return generic 500 to client
        alog.exception("Prediction error")
        raise HTTPException(500, f"Internal error: {e}")
