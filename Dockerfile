# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY app app
COPY model model
COPY reports reports

# expose & healthcheck
EXPOSE 8000
# periodically call GET /health; if it fails, container is marked unhealthy
HEALTHCHECK CMD curl -fsS http://localhost:8000/health || exit 1

# Default command to start the FastAPI app using uvicorn
# - app.main:app  -> module:object (FastAPI instance)
# - host 0.0.0.0 -> listen on all interfaces
# - port 8000    -> match EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port","8000"]
