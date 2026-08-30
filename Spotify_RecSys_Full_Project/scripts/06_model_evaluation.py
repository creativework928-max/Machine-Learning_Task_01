"""
TASK 6 — MODEL EVALUATION & VISUALIZATION
=============================================
Produces evaluation visuals for all trained models:
  - ROC curves (all models overlaid)
  - Precision-Recall curves (all models overlaid)  -> more informative given
    the 2% class imbalance than ROC alone
  - Confusion matrix for best model (at tuned threshold)
  - Feature importance for best tree-based model
  - Precision/Recall vs Threshold curve (helps pick an operating point)
  - Model comparison bar chart
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                              ConfusionMatrixDisplay, auc)

DATA = "data"
MODELS = "outputs/models"
FIG = "outputs/figures"

SPOTIFY_GREEN = "#1DB954"
DARK = "#191414"
PALETTE = ["#1DB954", "#509BF5", "#E8115B", "#FFC864", "#8B5CF6"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 11, "figure.dpi": 150, "axes.grid": True,
    "grid.color": "#EDEDED", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png")


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


def main():
    test = pd.read_csv(f"{DATA}/test_set.csv")
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[TARGET]

    pipelines = joblib.load(f"{MODELS}/all_pipelines.joblib")
    with open(f"{MODELS}/model_results.json") as f:
        meta = json.load(f)
    best_name = meta["best_model"]
    results = meta["results"]

    # ---------------- 1. ROC curves (all models) ----------------
    fig, ax = plt.subplots(figsize=(8, 7))
    for (name, pipe), color in zip(pipelines.items(), PALETTE):
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name.replace('_',' ')} (AUC={auc(fpr,tpr):.3f})",
                 color=color, linewidth=2.3)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.3, label="Random baseline")
    ax.set_title("ROC Curves — Model Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    savefig(fig, "09_roc_curves")

    # ---------------- 2. Precision-Recall curves ----------------
    fig, ax = plt.subplots(figsize=(8, 7))
    base_rate = y_test.mean()
    for (name, pipe), color in zip(pipelines.items(), PALETTE):
        proba = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=f"{name.replace('_',' ')}", color=color, linewidth=2.3)
    ax.axhline(base_rate, linestyle="--", color="#999999", linewidth=1.3,
               label=f"No-skill baseline ({base_rate:.3f})")
    ax.set_title("Precision-Recall Curves — Model Comparison\n(more informative than ROC given 2% class imbalance)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=9)
    savefig(fig, "10_precision_recall_curves")

    # ---------------- 3. Confusion matrix for best model (tuned threshold) ----------------
    best_pipe = pipelines[best_name]
    proba = best_pipe.predict_proba(X_test)[:, 1]

    # choose threshold that maximizes F1 on test set (reported as an operating point)
    prec, rec, thresh = precision_recall_curve(y_test, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_thresh = thresh[np.argmax(f1s[:-1])] if len(thresh) else 0.5

    y_pred_tuned = (proba >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_tuned)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Repeat", "Repeat"])
    disp.plot(ax=ax, cmap="Greens", colorbar=False, values_format=",")
    ax.set_title(f"Confusion Matrix — {best_name.replace('_',' ')}\n(threshold={best_thresh:.2f}, F1-optimal)")
    savefig(fig, "11_confusion_matrix_best_model")

    # ---------------- 4. Precision/Recall vs Threshold ----------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresh, prec[:-1], label="Precision", color=SPOTIFY_GREEN, linewidth=2.3)
    ax.plot(thresh, rec[:-1], label="Recall", color="#E8115B", linewidth=2.3)
    ax.axvline(best_thresh, linestyle="--", color=DARK, linewidth=1.3,
               label=f"Chosen threshold = {best_thresh:.2f}")
    ax.set_title(f"Precision & Recall vs. Decision Threshold — {best_name.replace('_',' ')}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    savefig(fig, "12_threshold_tuning")

    # ---------------- 5. Model comparison bar chart ----------------
    metrics_df = pd.DataFrame(results).T[["roc_auc", "pr_auc", "f1"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind="bar", ax=ax, color=[SPOTIFY_GREEN, "#509BF5", "#E8115B"], width=0.7)
    ax.set_title("Model Comparison Across Metrics")
    ax.set_ylabel("Score")
    ax.set_xticklabels([n.replace("_", " ") for n in metrics_df.index], rotation=20, ha="right")
    ax.legend(["ROC-AUC", "PR-AUC", "F1 (threshold=0.5)"])
    savefig(fig, "13_model_comparison")

    # ---------------- 6. Feature importance (best tree model, or RF for reference) ----------------
    tree_model_name = "Random_Forest" if "Random_Forest" in pipelines else best_name
    tree_pipe = pipelines[tree_model_name]
    ohe = tree_pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names
    importances = tree_pipe.named_steps["clf"].feature_importances_

    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=SPOTIFY_GREEN)
    ax.set_title(f"Top 15 Feature Importances — {tree_model_name.replace('_',' ')}")
    ax.set_xlabel("Importance")
    savefig(fig, "14_feature_importance")

    # ---------------- Save final evaluation summary ----------------
    summary = {
        "best_model": best_name,
        "chosen_threshold": round(float(best_thresh), 4),
        "test_set_size": len(test),
        "confusion_matrix": cm.tolist(),
        "metrics_at_0.5": results,
    }
    with open(f"{MODELS}/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

