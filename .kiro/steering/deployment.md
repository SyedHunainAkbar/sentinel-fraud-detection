---
inclusion: always
---

# Deployment Steering

## FastAPI scoring service (containerized)

- Base image: `python:3.11-slim`. Multi-stage build if needed to keep image small.
- Install only production dependencies (no dev/test extras).
- Copy `src/sentinel/` and `requirements.txt` into the image.
- Model artifacts (`models/*.joblib`) are mounted at runtime or fetched from object
  storage — never baked into the image.
- Entry point: `uvicorn sentinel.serving.api:app --host 0.0.0.0 --port 8000`.
- Health check: `GET /health` returns `{"status": "ok"}` (used by orchestrators).

## Streamlit executive dashboard

- **Streamlit Community Cloud**: entry point is `streamlit_app.py` at repo root, which
  imports and runs `src/sentinel/serving/dashboard.py`.
- For local/container runs: `streamlit run src/sentinel/serving/dashboard.py --server.port 8501`.
- The dashboard reads from `reports/` JSON files only — no model loading, no DB calls.

## Environment variables (no secrets in the image)

| Variable           | Purpose                                  | Required |
|--------------------|------------------------------------------|----------|
| `SENTINEL_DATA`    | Path to full transaction CSV             | No (defaults to sample) |
| `MODEL_PATH`       | Override model artifact directory         | No (defaults to `models/`) |
| `REVIEW_COST`      | Override per-alert cost at runtime        | No (defaults to config) |

- Secrets (API keys, DB credentials) are injected via environment or a secrets manager at
  deploy time. Never committed, never in the Dockerfile.

## Container hygiene

- Run as non-root user (`USER app`).
- Pin base image digest or minor version for reproducibility.
- `.dockerignore` excludes `data/`, `.git/`, `__pycache__/`, `reports/`, `models/`.
- No `COPY . .` — explicitly copy only what's needed.

## Health checks

- FastAPI: `GET /health` — returns 200 when model is loaded and ready to score.
- Streamlit: default Streamlit healthcheck on `/_stcore/health`.
- Container orchestrator (ECS/K8s): point liveness/readiness probes at these endpoints.
