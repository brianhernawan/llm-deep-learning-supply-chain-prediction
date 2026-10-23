# Carrier Advisor: predict freight cost, pick the cheapest carrier

**DSML Batch 42 Take Home Test · Brian Hernawan · US Logistics Performance dataset (2,000 shipments, 2023)**

A coordinator asks *"Ship 25 kg from LA to Boston, cheapest option"*. The advisor prices the shipment with
all 7 carriers using a trained model and recommends the cheapest. All costs are assumed USD.

## Results (`models/metrics.json`, 389 untouched test shipments)
| Model | Test MAE | R² |
|---|---|---|
| Linear Regression (baseline) | USD 16.68 | 0.922 |
| XGBoost, tuned | USD 13.54 | 0.936 |
| Keras MLP (TensorFlow) | USD 14.55 | 0.933 |
| **Average of XGBoost + Keras (served by default)** | **USD 13.12** | **0.939** |

- The served model is a fixed 50/50 average of tuned XGBoost and the Keras network (not tuned on the test set):
  21% lower error than linear regression; half of all quotes are within USD 9.38. Either model can still be picked alone.
- Score the average on your own machine: `python scripts/score_ensemble.py` (no retraining; adds it to metrics.json).
- Carriers charge significantly different rates per mile (Kruskal-Wallis p ≈ 1.6e-29): USPS ≈ USD 141 vs DHL ≈ USD 171 for 1,000 miles and 20 kg.
- Booking the cheapest predicted carrier every time: up to **USD 30,321 (7.8%)** saved on 1,955 shipments (upper bound).
- Delays can't be predicted from booking data (ROC-AUC ≈ 0.52), and reliability gaps between carriers are not significant (p ≈ 0.36).
- The data looks computer-generated (distance is not geographic), so distance is an input; the app uses the lane's median.

## Files
```
logistics-carrier-advisor/
├── data/                 # CSV + logistics.db (built by data.py)
├── notebooks/            # 01_data_eda.ipynb, 02_models.ipynb (executed)
├── models/               # xgb_cost.joblib, keras_cost.keras, keras_parts.joblib, metrics.json
├── testing/test_project.py
├── scripts/make_stack.py # builds the Docker stack folder
├── scripts/score_ensemble.py # test score + speed of the XGBoost + Keras average
├── data.py               # SQLite, cleaning rules, preprocessing pipeline
├── train.py              # baselines, tuned XGBoost, Keras, 5-fold CV, MLflow
├── predict.py            # price all carriers for one shipment (ensemble, xgboost or keras)
├── agent.py              # LangChain agent: quote tool + read-only SQL tool (Groq)
├── api.py                # FastAPI: /health /quote /ask
├── app.py                # Streamlit, one page: KPIs, carrier quote, ask the advisor, carrier charts
└── requirements.txt
```

## Rubric mapping
| Rubric item | Where |
|---|---|
| Problem statement | this README, deck |
| Preprocessing + EDA | `data.py` (SQL view, cleaning, pipeline), notebook 01 (Kruskal-Wallis, chi-square, Spearman) |
| ML / DL architecture | `train.py`: Linear, Decision Tree, Random Forest, XGBoost, TensorFlow MLP |
| Evaluation + tuning | 5-fold CV for every model, RandomizedSearchCV (30 × 5 folds), 20% hold-out test, notebook 02 |
| Deployment + MLOps | Streamlit, FastAPI, Docker stack, MLflow tracking of every training run |
| Business insight | notebook 02: rate card, savings simulation, recommendations |
| Advanced integration (bonus) | LangChain agent calling the model and SQL through the API |

## Run it
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # put your Groq key in .env
python data.py                     # build the database
python train.py                    # ~3-5 min, writes models/ and mlflow.db
python scripts/score_ensemble.py   # scores the served average, adds it to metrics.json
python -m testing.test_project     # expected: 7 checks passed
```
In tmux, one window each (`Ctrl+b c` for a new window; activate the venv in every window):
```bash
uvicorn api:app --host 0.0.0.0 --port 8010
API_URL=http://localhost:8010 streamlit run app.py --server.port 8510
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5010
```
App: http://localhost:8510 · API docs: http://localhost:8010/docs · MLflow: http://localhost:5010 (click "Model training").

## Docker
```bash
python scripts/make_stack.py       # creates ~/docker_volumes/logistics-carrier-advisor/
cd ~/docker_volumes/logistics-carrier-advisor && cp .env.example .env   # add the key
docker compose build && docker compose up -d   # app on 8510, API on 8010
```

## Limits
Computer-generated data (distance is not geographic); savings ignore carrier capacity; no delay model; no login on the API; retraining is manual.
