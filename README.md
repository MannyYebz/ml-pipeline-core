# ml-pipeline-core

Production-style end-to-end machine learning pipeline for **Telco Customer Churn** prediction.

- **Model**: PyTorch feedforward neural network (3+ hidden layers, batch norm, dropout, sigmoid head)
- **Tracking**: MLflow experiment tracking + Model Registry
- **Serving**: FastAPI with `/predict` and `/health` endpoints
- **Deploy**: Dockerized, ready for Render via `render.yaml`
- **Tests**: `pytest` for preprocessing and model components

---

## Architecture

```
                  +------------------+
                  |   Kaggle API /   |
                  |  Public Mirror   |
                  +---------+--------+
                            |
                            v
                  +---------+--------+
                  |   data/fetch.py  |   raw CSV  -> data/raw/telco_churn.csv
                  +---------+--------+
                            |
                            v
        +-------------------+--------------------+
        |              src/preprocess.py         |
        |  - stratified train/val/test split     |
        |  - impute, scale (numeric)             |
        |  - impute, one-hot (categorical)       |
        |  - Preprocessor saved with joblib  ----+----> artifacts/preprocessor.joblib
        +-------------------+--------------------+
                            |
                            v
        +-------------------+--------------------+        +----------------------+
        |    src/dataset.py  +  src/model.py     |        |     MLflow Server    |
        |       PyTorch Dataset + nn.Module      |        |  - experiment runs   |
        +-------------------+--------------------+        |  - artifacts         |
                            |                             |  - model registry    |
                            v                             +----------+-----------+
        +-------------------+--------------------+                   ^
        |               src/train.py             |   log per-epoch   |
        |  - loss / AUC-ROC every epoch          +-------------------+
        |  - early stopping on val AUC           |
        |  - best checkpoint saved (*.pt)        |
        |  - src/registry.py registers + stages  |
        +-------------------+--------------------+
                            |
                            v
        +-------------------+--------------------+
        |               src/evaluate.py          |   accuracy / AUC / F1 / CM
        +-------------------+--------------------+
                            |
                            v
        +-------------------+--------------------+
        |                  api/                  |
        |  - schema.py     Pydantic request/resp |
        |  - load_model.py Loads Production from |
        |                  MLflow Registry       |
        |  - main.py       FastAPI app           |
        +-------------------+--------------------+
                            |
                            v
                      Render (Docker)
```

---

## Project layout

```
ml-pipeline-core/
├── data/
│   └── fetch.py
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── registry.py
├── api/
│   ├── main.py
│   ├── schema.py
│   └── load_model.py
├── tests/
│   ├── test_preprocess.py
│   └── test_model.py
├── Dockerfile
├── render.yaml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # then edit values
export PYTHONPATH=$(pwd)            # Windows: set PYTHONPATH=%cd%
```

Kaggle credentials are optional — `data/fetch.py` falls back to a public mirror if
they are not configured.

---

## End-to-end pipeline

```bash
# 1. Fetch the dataset
python -m data.fetch

# 2. Train, log to MLflow, register, promote to Staging
python -m src.train

# 2a. (Optional) promote directly to Production
python -m src.train --promote-to-production

# 3. Inspect the MLflow UI
mlflow ui --backend-store-uri ./mlruns

# 4. Serve the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Example request

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 1024.40
  }'
```

```json
{ "churn": true, "probability": 0.7421 }
```

---

## Tests

```bash
pytest -q
```

The suite uses a synthetic Telco-shaped dataframe (see `tests/conftest.py`) so it
runs without downloading the dataset.

---

## Configuration

Every hyperparameter lives in [`src/config.py`](src/config.py). Override at
runtime via environment variables — see [`.env.example`](.env.example) for the
full list.

Important defaults:

| Setting           | Default              | Env var                    |
|-------------------|----------------------|----------------------------|
| Learning rate     | `1e-3`               | `LEARNING_RATE`            |
| Batch size        | `64`                 | `BATCH_SIZE`               |
| Epochs            | `30`                 | `EPOCHS`                   |
| Early stop patience | `5`                | `PATIENCE`                 |
| Hidden dims       | `[128, 64, 32]`      | _(set in `ModelConfig`)_   |
| Dropout           | `0.3`                | _(set in `ModelConfig`)_   |
| MLflow tracking   | `file:./mlruns`      | `MLFLOW_TRACKING_URI`      |
| Model stage       | `Production`         | `API_MODEL_STAGE`          |

---

## Deploying to Render

1. Push the repository to GitHub.
2. In the Render dashboard, **New +** → **Blueprint** and point it at your repo.
   Render reads [`render.yaml`](render.yaml) and provisions a Docker web service.
3. Set `MLFLOW_TRACKING_URI` in the Render dashboard to your remote tracking
   server (e.g. an `https://...` Databricks workspace or a self-hosted MLflow
   instance). The local `file:./mlruns` default will not survive across deploys.
4. Make sure a registered model version exists at the `API_MODEL_STAGE` you
   configured (default `Production`). Train and promote before the first deploy:
   ```bash
   python -m src.train --promote-to-production
   ```
5. The service exposes `/health` (used by Render's health check) and `/predict`.

### Local Docker

```bash
docker build -t ml-pipeline-core .
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=file:/app/mlruns \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/artifacts:/app/artifacts \
  ml-pipeline-core
```

---

## Design notes

- **No data leakage**: the preprocessor is fit on the training split only and is
  persisted to `artifacts/preprocessor.joblib`. The API reloads the same artifact
  so inference-time features match the training-time distribution.
- **Reproducibility**: seeds are set in `src/train.py`, splits are stratified, and
  every training run records its parameters and metrics to MLflow.
- **Separation of concerns**: configuration, data, modeling, training,
  evaluation, registry, and serving each live in their own module with explicit
  typed boundaries.
