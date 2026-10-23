"""Score the served ensemble (XGBoost + Keras average) on the same 389 test rows as train.py.
Uses the saved models, so no retraining. Run from the project root:  python scripts/score_ensemble.py
Adds "Ensemble (XGBoost + Keras)" to models/metrics.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split  # noqa: E402

from data import FEATURES, TARGET, fit_rules, load_shipments, training_rows  # noqa: E402
from predict import MODELS_DIR, predict_cost  # noqa: E402
from train import SEED, test_scores  # noqa: E402


def main():
    df = load_shipments()
    train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=SEED)  # same split as train.py
    test = training_rows(test_raw, fit_rules(train_raw))
    X, y = test[FEATURES], test[TARGET]
    predict_cost(X.head(7), "ensemble")  # warm-up: first call loads the models
    scores = {}
    for model in ["xgboost", "keras", "ensemble"]:
        start = time.perf_counter()
        pred = predict_cost(X.head(7), model)  # one quote = 7 carrier rows
        ms = 1000 * (time.perf_counter() - start)
        scores[model] = {**test_scores(y, predict_cost(X, model)), "ms_per_quote": round(ms, 1)}
        print(f"{model:9s} MAE {scores[model]['mae']:.2f}  median {scores[model]['median_abs_error']:.2f}  "
              f"R2 {scores[model]['r2']:.3f}  {ms:.0f} ms per quote ({len(pred)} carriers)")
    path = MODELS_DIR / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["test"]["Ensemble (XGBoost + Keras)"] = scores["ensemble"]
    path.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
