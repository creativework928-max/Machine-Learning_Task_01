"""
TASK 4 — FEATURE ENGINEERING
================================
Builds the final model-ready feature matrix by combining:
  - Song audio features
  - User behavioral aggregates
  - User-song interaction context features
  - Temporal features

Output: /data/model_features.csv  (one row per (user,song) interaction)
"""

import pandas as pd
import numpy as np

DATA = "data"


def main():
    users = pd.read_csv(f"{DATA}/users_clean.csv", parse_dates=["signup_date"])
    songs = pd.read_csv(f"{DATA}/songs_clean.csv", parse_dates=["release_date"])
    history = pd.read_csv(f"{DATA}/listening_history_clean.csv", parse_dates=["timestamp"])
    interactions = pd.read_csv(f"{DATA}/interactions.csv", parse_dates=["first_listen_ts"])

    # ---------------------------------------------------------
    # A. USER-LEVEL behavioral features (computed from full history)
    # ---------------------------------------------------------
    user_feats = history.groupby("user_id").agg(
        user_total_plays=("song_id", "count"),
        user_unique_songs=("song_id", "nunique"),
        user_avg_ms_played_ratio=("ms_played_ratio", "mean"),
        user_skip_rate=("skipped", "mean"),
    ).reset_index()
    user_feats["user_repeat_tendency"] = (
        user_feats["user_total_plays"] / user_feats["user_unique_songs"]
    )  # avg plays per unique song -> how "repeat-happy" this user generally is

    # user genre diversity (entropy-like: unique genres / total plays)
    genre_hist = history.merge(songs[["song_id", "genre"]], on="song_id")
    genre_div = genre_hist.groupby("user_id")["genre"].nunique().rename("user_genre_diversity")
    user_feats = user_feats.merge(genre_div, on="user_id", how="left")

    # ---------------------------------------------------------
    # B. SONG-LEVEL popularity-derived features
    # ---------------------------------------------------------
    song_stats = history.groupby("song_id").agg(
        song_total_plays=("user_id", "count"),
        song_unique_listeners=("user_id", "nunique"),
        song_avg_skip_rate=("skipped", "mean"),
    ).reset_index()

    # ---------------------------------------------------------
    # C. MERGE everything into the interaction table
    # ---------------------------------------------------------
    df = interactions.merge(songs, on="song_id", how="left")
    df = df.merge(users[["user_id", "signup_date", "subscription_tier",
                          "primary_device", "country"]], on="user_id", how="left")
    df = df.merge(user_feats, on="user_id", how="left")
    df = df.merge(song_stats, on="song_id", how="left")

    # ---------------------------------------------------------
    # D. Temporal / derived features
    # ---------------------------------------------------------
    df["days_since_release_at_first_listen"] = (
        df["first_listen_ts"] - df["release_date"]
    ).dt.days.clip(lower=0)
    df["user_tenure_days_at_first_listen"] = (
        df["first_listen_ts"] - df["signup_date"]
    ).dt.days.clip(lower=0)
    df["first_listen_hour"] = df["first_listen_ts"].dt.hour
    df["first_listen_dow"] = df["first_listen_ts"].dt.dayofweek
    df["first_listen_is_weekend"] = df["first_listen_dow"].isin([5, 6]).astype(int)

    # song "freshness" flag
    df["is_new_release"] = (df["days_since_release_at_first_listen"] <= 30).astype(int)

    # engagement quality on first exposure ONLY (leakage-safe: uses first play, not repeats)
    df["strong_first_listen"] = (df["first_listen_ms_played_ratio"] >= 0.8).astype(int)

    # ---------------------------------------------------------
    # E. Final feature selection
    # ---------------------------------------------------------
    feature_cols = [
        # song audio features
        "danceability", "energy", "valence", "acousticness", "instrumentalness",
        "liveness", "speechiness", "tempo", "loudness_db", "popularity", "duration_ms",
        # song popularity stats
        "song_total_plays", "song_unique_listeners", "song_avg_skip_rate",
        # user behavior
        "user_total_plays", "user_unique_songs", "user_avg_ms_played_ratio",
        "user_skip_rate", "user_repeat_tendency", "user_genre_diversity",
        # interaction-specific — FIRST LISTEN ONLY (leakage-safe, no future info)
        "first_listen_ms_played_ratio", "first_listen_skipped",
        "days_since_release_at_first_listen", "user_tenure_days_at_first_listen",
        "first_listen_hour", "first_listen_dow", "first_listen_is_weekend",
        "is_new_release", "strong_first_listen",
        # categorical
        "genre", "subscription_tier", "primary_device", "first_listen_context",
    ]
    # NOTE: total_play_count is deliberately EXCLUDED — it is computed over the
    # user's entire play history for this song, so a value >1 near-perfectly
    # encodes the label itself (classic target leakage). It is retained in
    # interactions.csv for reporting/analysis only, never as a model feature.
    target_col = "repeat_within_30_days"

    model_df = df[["user_id", "song_id"] + feature_cols + [target_col]].copy()
    model_df = model_df.dropna()

    print(f"Final feature matrix shape: {model_df.shape}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Target positive rate: {model_df[target_col].mean():.3%}")

    model_df.to_csv(f"{DATA}/model_features.csv", index=False)
    print(f"\nSaved: {DATA}/model_features.csv")


if __name__ == "__main__":
    main()

