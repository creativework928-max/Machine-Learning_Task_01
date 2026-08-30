"""
TASK 5 — MODEL TRAINING
===========================
Trains and compares 4 classifiers to predict repeat_within_30_days:
    1. Logistic Regression (baseline, interpretable)
    2. Random Forest
    3. Gradient Boosting (sklearn's boosted trees - XGBoost-equivalent)
    4. Random Forest with class_weight='balanced' (imbalance-aware)

Uses a TIME-BASED split (train on earlier interactions, test on later ones)
to avoid temporal leakage, since this is a real-world time-dependent task.

Saves: trained models, scaler/encoders, and metrics to outputs/models/
"""

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, classification_report)

DATA = "data"
MODELS = "outputs/models"

NUMERIC_FEATURES = [
    "danceability", "energy", "valence", "acousticness", "instrumentalness",
    "liveness", "speechiness", "tempo", "loudness_db", "popularity", "duration_ms",
    "song_total_plays", "song_unique_listeners", "song_avg_skip_rate",
    "user_total_plays", "user_unique_songs", "user_avg_ms_played_ratio",
    "user_skip_rate", "user_repeat_tendency", "user_genre_diversity",
    "first_listen_ms_played_ratio", "first_listen_skipped",
    "days_since_release_at_first_listen", "user_tenure_days_at_first_listen",
    "first_listen_hour", "first_listen_dow", "first_listen_is_weekend",
    "is_new_release", "strong_first_listen",
]
CATEGORICAL_FEATURES = ["genre", "subscription_tier", "primary_device", "first_listen_context"]
TARGET = "repeat_within_30_days"


def time_based_split(df, interactions_meta, test_frac=0.2):
    """Split by first_listen_ts so test set is strictly later in time than train."""
    df = df.merge(interactions_meta[["user_id", "song_id", "first_listen_ts"]],
                   on=["user_id", "song_id"])
    df = df.sort_values("first_listen_ts")
    cutoff_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:cutoff_idx].drop(columns=["first_listen_ts"])
    test = df.iloc[cutoff_idx:].drop(columns=["first_listen_ts"])
    return train, test


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(model_name, y_true, y_pred, y_proba, results):
    results[model_name] = {
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }
    print(f"\n--- {model_name} ---")
    for k, v in results[model_name].items():
        print(f"  {k}: {v}")


def main():
    df = pd.read_csv(f"{DATA}/model_features.csv")
    interactions_meta = pd.read_csv(f"{DATA}/interactions.csv", parse_dates=["first_listen_ts"])

    train, test = time_based_split(df, interactions_meta, test_frac=0.2)
    print(f"Train size: {len(train)} | Test size: {len(test)}")
    print(f"Train positive rate: {train[TARGET].mean():.3%} | Test positive rate: {test[TARGET].mean():.3%}")

    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET]
    X_test, y_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[TARGET]

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=42),
        "Random_Forest_Balanced": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.08, random_state=42),
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        evaluate(name, y_test, y_pred, y_proba, results)
        fitted_pipelines[name] = pipe

    # pick best by ROC-AUC (robust to imbalance + threshold-independent)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n>>> BEST MODEL: {best_name} (ROC-AUC={results[best_name]['roc_auc']})")

    # save all fitted pipelines + results + test set for Task 6 evaluation
    joblib.dump(fitted_pipelines, f"{MODELS}/all_pipelines.joblib")
    joblib.dump(fitted_pipelines[best_name], f"{MODELS}/best_pipeline.joblib")
    with open(f"{MODELS}/model_results.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)
    test.to_csv(f"{DATA}/test_set.csv", index=False)
    train.to_csv(f"{DATA}/train_set.csv", index=False)

    print(f"\nSaved models & results to {MODELS}")


if __name__ == "__main__":
    main()

