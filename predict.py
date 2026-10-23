"""Price one shipment with every carrier, using the saved models.

Used by the API (and through it, the app and the LLM agent).
"""
import os
from datetime import date
from pathlib import Path
from functools import lru_cache

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from data import FEATURES, PROJECT_DIR, clean_features, run_query  # noqa: E402

MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_DIR / "models"))


@lru_cache(maxsize=1)
def load_models():
    """Load once, then reuse: XGBoost pipeline, Keras network and the cleaning rules."""
    import json
    import tensorflow as tf
    return {"xgboost": joblib.load(MODELS_DIR / "xgb_cost.joblib"),
            "keras": tf.keras.models.load_model(MODELS_DIR / "keras_cost.keras"),
            "keras_parts": joblib.load(MODELS_DIR / "keras_parts.joblib"),
            "rules": json.loads((MODELS_DIR / "metrics.json").read_text())["cleaning_rules"]}


@lru_cache(maxsize=1)
def options():
    """Valid warehouses, cities and carriers, read from the database."""
    df = run_query("SELECT DISTINCT Origin_Warehouse, Destination, Carrier FROM shipments_clean")
    return {col: sorted(df[col].unique()) for col in df.columns}


def lane_distance(origin, destination):
    """Median past distance on this lane (the dataset's distance is not geographic)."""
    df = run_query("SELECT Distance_miles FROM shipments_clean WHERE Origin_Warehouse = ? AND Destination = ?",
                   params=(origin, destination))
    return float(df["Distance_miles"].median()) if len(df) else 1263.0


def check_inputs(origin, destination, weight_kg):
    if origin not in options()["Origin_Warehouse"]:
        raise ValueError(f"Unknown origin '{origin}'. Use one of {options()['Origin_Warehouse']}")
    if destination not in options()["Destination"]:
        raise ValueError(f"Unknown destination '{destination}'. Use one of {options()['Destination']}")
    if not weight_kg or weight_kg <= 0:
        raise ValueError("Weight must be more than 0 kg.")


def predict_cost(X, model="ensemble"):
    """Cost in USD. 'ensemble' = simple average of XGBoost and the Keras network (fixed 50/50, not tuned)."""
    m = load_models()
    xgb = m["xgboost"].predict(X)
    if model == "xgboost":
        return xgb
    raw = m["keras"].predict(m["keras_parts"]["prep"].transform(X), verbose=0)
    keras = m["keras_parts"]["y_scaler"].inverse_transform(raw).ravel()
    return keras if model == "keras" else (xgb + keras) / 2


def quote(origin, destination, weight_kg, distance_miles=None, ship_date=None, model="ensemble"):
    """Predicted cost (USD) for all 7 carriers, cheapest first, with past exception rate."""
    check_inputs(origin, destination, weight_kg)
    ship_date = pd.Timestamp(ship_date or date.today())
    rows = pd.DataFrame({"Carrier": options()["Carrier"]})
    rows["Origin_Warehouse"], rows["Destination"] = origin, destination
    rows["Weight_kg"] = float(weight_kg)
    rows["Distance_miles"] = float(distance_miles or lane_distance(origin, destination))
    rows["ship_month"], rows["ship_dayofweek"] = ship_date.month, ship_date.dayofweek
    X = clean_features(rows, load_models()["rules"])[FEATURES]
    cost = predict_cost(X, model)
    history = run_query("SELECT Carrier, ROUND(100.0 * AVG(is_exception), 1) AS past_exception_pct "
                        "FROM shipments_clean WHERE Status != 'In Transit' GROUP BY Carrier")
    result = pd.DataFrame({"Carrier": rows["Carrier"], "predicted_cost_usd": cost.astype(float).round(2)})
    result = result.merge(history, on="Carrier").sort_values("predicted_cost_usd")
    result["distance_miles"] = rows["Distance_miles"].iloc[0]
    return result.reset_index(drop=True)
