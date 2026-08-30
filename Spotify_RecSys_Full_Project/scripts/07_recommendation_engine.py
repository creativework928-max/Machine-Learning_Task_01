"""
TASK 7 — RECOMMENDATION ENGINE
==================================
Turns the trained repeat-listen probability model into an actual
personalized recommendation system:

  For a given user:
    1. Build CANDIDATE songs (songs the user hasn't already played,
       biased toward their favorite genres + globally popular tracks
       — a simple "candidate generation" stage, same idea Spotify uses
       before ranking).
    2. Score every candidate with P(repeat_within_30_days) from the
       best trained model — used here as a proxy for "this user will
       love this song enough to come back to it".
    3. Apply diversity re-ranking (avoid recommending 10 songs from the
       same artist).
    4. Return Top-N ranked recommendations.

Outputs:
  - outputs/reports/sample_recommendations.csv (for 10 sample users)
  - outputs/figures/15_sample_user_recommendations.png (visual example)
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

DATA = "data"
MODELS = "outputs/models"
REPORTS = "outputs/reports"
FIG = "outputs/figures"

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


def build_candidate_features(user_id, users, songs, history, user_agg, song_agg, top_k_candidates=200):
    """Construct a feature row per candidate song for a given user, mimicking
    what the feature engineering stage (Task 4) would produce at inference time."""
    user_row = users[users.user_id == user_id].iloc[0]
    played_songs = set(history.loc[history.user_id == user_id, "song_id"])

    candidates = songs[~songs.song_id.isin(played_songs)].copy()

    # candidate generation: favor user's favorite genres (approximate using their
    # historically most-played genres) + overall popularity, then sample a pool
    user_hist = history[history.user_id == user_id].merge(songs[["song_id", "genre"]], on="song_id")
    top_genres = user_hist["genre"].value_counts().head(3).index.tolist()
    if top_genres:
        weighted = candidates.copy()
        weighted["is_fav_genre"] = weighted["genre"].isin(top_genres).astype(int)
        weighted = weighted.sort_values(["is_fav_genre", "popularity"], ascending=[False, False])
        candidates = weighted.head(top_k_candidates)
    else:
        candidates = candidates.sort_values("popularity", ascending=False).head(top_k_candidates)

    n = len(candidates)
    feat = candidates.copy()

    # user-level features (constant across candidates for this user)
    urow = user_agg[user_agg.user_id == user_id]
    for col in ["user_total_plays", "user_unique_songs", "user_avg_ms_played_ratio",
                "user_skip_rate", "user_repeat_tendency", "user_genre_diversity"]:
        feat[col] = urow[col].values[0] if len(urow) else 0

    feat["subscription_tier"] = user_row["subscription_tier"]
    feat["primary_device"] = user_row["primary_device"]

    # song-level popularity stats
    feat = feat.merge(song_agg, on="song_id", how="left")
    feat[["song_total_plays", "song_unique_listeners", "song_avg_skip_rate"]] = \
        feat[["song_total_plays", "song_unique_listeners", "song_avg_skip_rate"]].fillna(0)

    # neutral assumptions for "first listen" behavioral features (unknown at
    # recommendation time -> use catalog-average / sensible defaults)
    feat["first_listen_ms_played_ratio"] = 0.7
    feat["first_listen_skipped"] = 0
    feat["first_listen_context"] = "recommended"
    feat["days_since_release_at_first_listen"] = (
        pd.Timestamp.today() - feat["release_date"]).dt.days.clip(lower=0)
    feat["user_tenure_days_at_first_listen"] = (
        pd.Timestamp.today() - pd.to_datetime(user_row["signup_date"])).days
    feat["first_listen_hour"] = pd.Timestamp.today().hour
    feat["first_listen_dow"] = pd.Timestamp.today().dayofweek
    feat["first_listen_is_weekend"] = int(feat["first_listen_dow"].iloc[0] in [5, 6])
    feat["is_new_release"] = (feat["days_since_release_at_first_listen"] <= 30).astype(int)
    feat["strong_first_listen"] = 0

    return feat


def recommend_for_user(user_id, users, songs, history, user_agg, song_agg, model, top_n=10):
    cand = build_candidate_features(user_id, users, songs, history, user_agg, song_agg)
    X = cand[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    cand["predicted_repeat_prob"] = model.predict_proba(X)[:, 1]

    # diversity re-ranking: cap at 2 songs per artist among the ranked results
    cand = cand.sort_values("predicted_repeat_prob", ascending=False)
    seen_artists = {}
    picked = []
    for _, row in cand.iterrows():
        a = row["artist_id"]
        if seen_artists.get(a, 0) >= 2:
            continue
        picked.append(row)
        seen_artists[a] = seen_artists.get(a, 0) + 1
        if len(picked) >= top_n:
            break

    return pd.DataFrame(picked)[
        ["song_id", "track_name", "artist_id", "genre", "popularity", "predicted_repeat_prob"]
    ].reset_index(drop=True)


def main():
    users = pd.read_csv(f"{DATA}/users_clean.csv", parse_dates=["signup_date"])
    songs = pd.read_csv(f"{DATA}/songs_clean.csv", parse_dates=["release_date"])
    history = pd.read_csv(f"{DATA}/listening_history_clean.csv", parse_dates=["timestamp"])
    model = joblib.load(f"{MODELS}/best_pipeline.joblib")

    user_agg = history.groupby("user_id").agg(
        user_total_plays=("song_id", "count"),
        user_unique_songs=("song_id", "nunique"),
        user_avg_ms_played_ratio=("ms_played_ratio", "mean"),
        user_skip_rate=("skipped", "mean"),
    ).reset_index()
    user_agg["user_repeat_tendency"] = user_agg["user_total_plays"] / user_agg["user_unique_songs"]
    genre_hist = history.merge(songs[["song_id", "genre"]], on="song_id")
    genre_div = genre_hist.groupby("user_id")["genre"].nunique().rename("user_genre_diversity")
    user_agg = user_agg.merge(genre_div, on="user_id", how="left")

    song_agg = history.groupby("song_id").agg(
        song_total_plays=("user_id", "count"),
        song_unique_listeners=("user_id", "nunique"),
        song_avg_skip_rate=("skipped", "mean"),
    ).reset_index()

    sample_users = user_agg.sort_values("user_total_plays", ascending=False)["user_id"].head(10).tolist()

    all_recs = []
    for uid in sample_users:
        recs = recommend_for_user(uid, users, songs, history, user_agg, song_agg, model, top_n=10)
        recs.insert(0, "user_id", uid)
        all_recs.append(recs)

    result = pd.concat(all_recs, ignore_index=True)
    result.to_csv(f"{REPORTS}/sample_recommendations.csv", index=False)
    print(f"Saved sample recommendations for {len(sample_users)} users -> {REPORTS}/sample_recommendations.csv")
    print(result.head(15).to_string(index=False))

    # ---- visualize recommendations for ONE example user ----
    example_uid = sample_users[0]
    example = result[result.user_id == example_uid].sort_values("predicted_repeat_prob")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(example)))
    ax.barh(example["track_name"] + "  (" + example["genre"] + ")",
            example["predicted_repeat_prob"] * 100, color=colors)
    ax.set_title(f"Top-10 Personalized Recommendations for User #{example_uid}\nRanked by Predicted Repeat-Listen Probability")
    ax.set_xlabel("Predicted probability of repeat listen (%)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/15_sample_user_recommendations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved 15_sample_user_recommendations.png")


if __name__ == "__main__":
    main()

