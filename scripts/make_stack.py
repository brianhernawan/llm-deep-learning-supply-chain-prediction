"""Build the Docker stack folder from this project (the release step).

Creates ~/docker_volumes/logistics-carrier-advisor/ with only what serving needs:
serving code in data/, trained weights in models/, the database in storage/.
Training files (notebooks, train.py, tests) stay here in the workspace.

Run from the project root (after python train.py):
    python scripts/make_stack.py            # default target folder
    python scripts/make_stack.py /some/dir  # custom target folder
Safe to run again: it replaces data/ and models/, and keeps .env.
"""
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "docker_volumes" / "logistics-carrier-advisor"

SERVING_FILES = ["api.py", "app.py", "predict.py", "data.py", "agent.py"]
MODEL_FILES = ["xgb_cost.joblib", "keras_cost.keras", "keras_parts.joblib", "metrics.json"]

DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
"""

REQUIREMENTS = """pandas==3.0.6
numpy==2.4.6
scipy==1.17.1
scikit-learn==1.9.1
xgboost==3.2.0
tensorflow-cpu==2.21.0
joblib==1.6.0
plotly==7.1.0
fastapi==0.143.3
uvicorn==0.54.0
requests==2.34.2
streamlit==1.66.0
langchain==1.4.6
langchain-core==1.6.11
langchain-groq==1.1.3
python-dotenv==1.2.4
"""

COMPOSE = """---
name: logistics-carrier-advisor

x-logging: &logging
  driver: json-file
  options:
    max-size: 10m
    max-file: "3"

services:
  api:
    build: ./data
    image: kwf/logistics-carrier-advisor:latest
    container_name: carrier-advisor-api
    command: uvicorn api:app --host 0.0.0.0 --port 8000
    env_file: .env
    environment:
      - TZ=Asia/Jakarta
      - DB_PATH=/app/storage/logistics.db
    volumes:
      - ./models:/app/models:ro
      - ./storage:/app/storage
    ports:
      - "8010:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    logging: *logging
    networks:
      - backend

  app:
    image: kwf/logistics-carrier-advisor:latest
    container_name: carrier-advisor-app
    command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
    env_file: .env
    environment:
      - TZ=Asia/Jakarta
      - API_URL=http://carrier-advisor-api:8000
      - DB_PATH=/app/storage/logistics.db
    volumes:
      - ./storage:/app/storage:ro
    ports:
      - "8510:8501"
    depends_on:
      api:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    logging: *logging
    networks:
      - frontend
      - backend

networks:
  frontend:
    external: true
  backend:
    external: true
"""

ENV_EXAMPLE = "GROQ_API_KEY=your-groq-key-here\nGROQ_MODEL=openai/gpt-oss-20b\n"

README = """# Carrier Advisor: Docker stack

Serving-only stack built by `python scripts/make_stack.py` from the training workspace.

## First run
```bash
cd ~/docker_volumes/logistics-carrier-advisor
cp .env.example .env                  # then put your real GROQ_API_KEY in .env
docker network create frontend 2>/dev/null || true
docker network create backend 2>/dev/null || true
docker compose build                  # about 5 minutes the first time (TensorFlow)
docker compose up -d
docker compose ps                     # both services should become "healthy"
```
- App: http://localhost:8510
- API docs: http://localhost:8010/docs

## Folder layout
| Path | Content |
|---|---|
| `compose.yaml` | two services (api, app) sharing one image |
| `data/` | build context: Dockerfile, requirements.txt, serving code |
| `models/` | trained weights, mounted read-only |
| `storage/` | SQLite database |

## Update after retraining
```bash
cd ~/logistics-carrier-advisor && python train.py && python scripts/make_stack.py
cd ~/docker_volumes/logistics-carrier-advisor && docker compose up -d --build
```
Downtime: about 10-20 seconds while the containers restart. Rollback: rerun make_stack.py
from the previous git commit of the workspace and `docker compose up -d --build`.

## Common errors
| Error | Fix |
|---|---|
| `network frontend declared as external, but could not be found` | run the two `docker network create` lines |
| Advisor tab says GROQ_API_KEY is not set | edit `.env`, then `docker compose up -d` |
| `429` from Groq | free-tier limit (6,000 tokens/min), so wait one minute |
"""


def copy_serving_code(data_dir):
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    for name in SERVING_FILES:
        shutil.copy(PROJECT / name, data_dir / name)
    (data_dir / "Dockerfile").write_text(DOCKERFILE)
    (data_dir / "requirements.txt").write_text(REQUIREMENTS)


def copy_models(models_dir):
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir()
    for name in MODEL_FILES:
        shutil.copy(PROJECT / "models" / name, models_dir / name)


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    copy_serving_code(TARGET / "data")
    copy_models(TARGET / "models")
    (TARGET / "storage").mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT / "data" / "logistics.db", TARGET / "storage" / "logistics.db")
    (TARGET / "compose.yaml").write_text(COMPOSE)
    (TARGET / ".env.example").write_text(ENV_EXAMPLE)
    (TARGET / "README.md").write_text(README)
    if not (TARGET / ".env").exists():
        print("Next: cp .env.example .env and add your GROQ_API_KEY")
    print(f"Stack ready at {TARGET}")


if __name__ == "__main__":
    main()
