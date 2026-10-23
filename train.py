"""Train the cost models, compare them, save the winners and log everything to MLflow.

Run from the project root:  python train.py      (about 3-5 minutes on a laptop CPU)
Numbers land in models/metrics.json: the single source for the README and the deck.
View runs: mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5010
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from data import FEATURES, PROJECT_DIR, TARGET, build_preprocessor, fit_rules, load_shipments, training_rows  # noqa: E402

SEED = 42
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)
MODELS_DIR = PROJECT_DIR / "models"


def make_pipeline(model):
    """Preprocessor + model together, so every CV fold refits the encoder (no leakage)."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def cv_mae(model, X, y):
    scores = cross_val_score(make_pipeline(model), X, y, cv=CV, scoring="neg_mean_absolute_error")
    return float(-scores.mean())


def compare_baselines(X, y):
    models = {"Mean baseline": DummyRegressor(), "Linear Regression": LinearRegression(),
              "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=SEED),
              "Random Forest": RandomForestRegressor(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)}
    return {name: cv_mae(model, X, y) for name, model in models.items()}


def tune_xgboost(X, y):
    """Random search: 30 settings x 5 folds, scored by MAE."""
    space = {"model__n_estimators": [200, 400, 600, 800], "model__max_depth": [2, 3, 4, 5, 6],
             "model__learning_rate": [0.01, 0.03, 0.05, 0.1], "model__subsample": [0.7, 0.85, 1.0],
             "model__colsample_bytree": [0.7, 0.85, 1.0], "model__min_child_weight": [1, 3, 5, 10],
             "model__reg_lambda": [0.1, 1.0, 5.0, 10.0]}
    search = RandomizedSearchCV(make_pipeline(XGBRegressor(random_state=SEED, n_jobs=-1)), space,
                                n_iter=30, cv=CV, scoring="neg_mean_absolute_error", random_state=SEED)
    search.fit(X, y)
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, params, float(-search.best_score_)


def build_network(n_inputs):
    """Keras MLP: 128 -> 64 -> 1 output (cost). Huber loss is robust to noisy prices."""
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="huber")
    return model


def fit_network(X, y):
    """Fit encoder, target scaler and network on the given rows. Returns all three."""
    prep, y_scaler = build_preprocessor(), StandardScaler()
    Xt = prep.fit_transform(X)
    yt = y_scaler.fit_transform(y.values.reshape(-1, 1))
    net = build_network(Xt.shape[1])
    stop = tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)
    history = net.fit(Xt, yt, validation_split=0.15, epochs=300, batch_size=32, callbacks=[stop], verbose=0)
    return net, prep, y_scaler, history.history


def predict_network(net, prep, y_scaler, X):
    return y_scaler.inverse_transform(net.predict(prep.transform(X), verbose=0)).ravel()


def cv_network(X, y):
    """5-fold CV for the network: each fold fits its own encoder and scaler."""
    maes = []
    for tr, va in CV.split(X):
        net, prep, y_scaler, _ = fit_network(X.iloc[tr], y.iloc[tr])
        maes.append(np.mean(np.abs(y.iloc[va] - predict_network(net, prep, y_scaler, X.iloc[va]))))
    return float(np.mean(maes))


def test_scores(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {"mae": float(np.mean(np.abs(err))),
            "median_abs_error": float(np.median(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))}


def log_run(name, params, metrics):
    """One MLflow child run per model."""
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("carrier-advisor")
    df = load_shipments()
    train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=SEED)
    rules = fit_rules(train_raw)
    train, test = training_rows(train_raw, rules), training_rows(test_raw, rules)
    X, y, Xt, yt = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    print(f"Train {len(train)} rows, test {len(test)} rows, caps {rules}")

    with mlflow.start_run(run_name="train"):
        baselines = compare_baselines(X, y)
        print("5-fold CV MAE (USD):", {k: round(v, 2) for k, v in baselines.items()})
        xgb, xgb_params, xgb_cv = tune_xgboost(X, y)
        print(f"Tuned XGBoost CV MAE {xgb_cv:.2f} with {xgb_params}")
        keras_cv = cv_network(X, y)
        print(f"Keras CV MAE {keras_cv:.2f}")
        net, prep, y_scaler, history = fit_network(X, y)

        linear = make_pipeline(LinearRegression()).fit(X, y)
        results = {"Linear Regression": test_scores(yt, linear.predict(Xt)),
                "XGBoost (tuned)": test_scores(yt, xgb.predict(Xt)),
                "Keras MLP": test_scores(yt, predict_network(net, prep, y_scaler, Xt))}
        for name, cv in baselines.items():
            log_run(name, {"stage": "baseline"}, {"cv_mae": cv})
        log_run("XGBoost (tuned)", xgb_params, {"cv_mae": xgb_cv, **{f"test_{k}": v for k, v in results["XGBoost (tuned)"].items()}})
        log_run("Keras MLP", {"layers": "128-64-1", "epochs": len(history["loss"])},
                {"cv_mae": keras_cv, **{f"test_{k}": v for k, v in results["Keras MLP"].items()}})

        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(xgb, MODELS_DIR / "xgb_cost.joblib")
        net.save(MODELS_DIR / "keras_cost.keras")
        joblib.dump({"prep": prep, "y_scaler": y_scaler}, MODELS_DIR / "keras_parts.joblib")
        metrics = {"rows": {"train": len(train), "test": len(test_raw), "test_used": len(yt)},
                   "cleaning_rules": rules, "cv_mae": {**baselines, "XGBoost (tuned)": xgb_cv, "Keras MLP": keras_cv},
                   "xgb_params": xgb_params, "keras_epochs": len(history["loss"]), "test": results}
        (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        (MODELS_DIR / "keras_history.json").write_text(json.dumps({k: [float(v) for v in vals] for k, vals in history.items()}))
        mlflow.log_artifacts(str(MODELS_DIR), artifact_path="models")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
