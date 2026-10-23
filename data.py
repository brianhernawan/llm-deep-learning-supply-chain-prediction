"""Data layer: build the SQLite database, read shipments, and the cleaning rules.

Build the database once (safe to run again):  python data.py
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "logistics_shipments_dataset.csv"
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_DIR / "data" / "logistics.db"))

# Inputs known at booking time. Delivery_Date and Status are outcomes: never inputs.
CATEGORICAL = ["Origin_Warehouse", "Destination", "Carrier"]
NUMERIC = ["Weight_kg", "Distance_miles", "ship_month", "ship_dayofweek"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Cost"

CLEAN_VIEW_SQL = """
CREATE VIEW shipments_clean AS
SELECT Shipment_ID, Origin_Warehouse, Destination, Carrier, Shipment_Date,
       CASE WHEN Delivery_Date < Shipment_Date THEN NULL ELSE Delivery_Date END AS Delivery_Date,
       CASE WHEN Weight_kg <= 0 THEN NULL ELSE Weight_kg END AS Weight_kg,
       Cost, Status, Distance_miles, Transit_Days,
       CASE WHEN Delivery_Date < Shipment_Date THEN 1 ELSE 0 END AS date_error,
       CASE WHEN Status = 'Delayed' THEN 1 ELSE 0 END AS is_delayed,
       CASE WHEN Status IN ('Delayed', 'Lost', 'Returned') THEN 1 ELSE 0 END AS is_exception
FROM shipments;
"""


def build_database():
    """CSV -> table 'shipments' + view 'shipments_clean'. Replaces both, so it is safe to rerun."""
    df = pd.read_csv(CSV_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("shipments", conn, if_exists="replace", index=False)
        conn.execute("DROP VIEW IF EXISTS shipments_clean")
        conn.execute(CLEAN_VIEW_SQL)
    print(f"Database ready: {DB_PATH} ({len(df):,} shipments)")


def run_query(sql, params=None):
    """Run a SELECT on a read-only connection and return a DataFrame."""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def run_safe_select(sql, max_rows=20):
    """For SQL written by the LLM: one SELECT only, small result."""
    text = sql.strip().rstrip(";").lower()
    if ";" in text or not text.startswith(("select", "with")):
        raise ValueError("Only one read-only SELECT query is allowed.")
    return run_query(sql).head(max_rows)


def load_shipments():
    """All shipments from the clean view, plus month and weekday of shipping."""
    df = run_query("SELECT * FROM shipments_clean")
    ship_date = pd.to_datetime(df["Shipment_Date"])
    df["ship_month"] = ship_date.dt.month
    df["ship_dayofweek"] = ship_date.dt.dayofweek
    return df


def iqr_upper_fence(series, k=3.0):
    """Q3 + 3 x IQR: flags only extreme values."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return float(q3 + k * (q3 - q1))


def fit_rules(train_df):
    """Outlier caps learned from TRAINING rows only (no leakage)."""
    return {"weight_cap_kg": iqr_upper_fence(train_df["Weight_kg"].dropna()),
            "cost_cap_usd": iqr_upper_fence(train_df["Cost"].dropna())}


def clean_features(df, rules):
    """Same input cleaning for training and for new bookings: cap extreme weights."""
    df = df.copy()
    df["Weight_kg"] = df["Weight_kg"].clip(upper=rules["weight_cap_kg"])
    return df


def training_rows(df, rules):
    """Rows the model learns from: cost present and not an extreme outlier."""
    df = df[df["Cost"].notna() & (df["Cost"] <= rules["cost_cap_usd"])]
    return clean_features(df, rules)


def build_preprocessor():
    """One-hot encode categories; fill missing numbers with the median, then scale."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", numeric, NUMERIC),
    ])


if __name__ == "__main__":
    build_database()
