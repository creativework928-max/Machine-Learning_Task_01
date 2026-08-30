"""
TASK 2 — DATA CLEANING & TARGET LABEL CONSTRUCTION
=====================================================
- Load raw tables from Task 1
- Clean / validate types, handle duplicates & missing values
- Build the (user, song) interaction table with the ML TARGET:
      repeat_within_30_days = 1  if the user played this song again
                                  within 30 days of their FIRST listen
                             = 0  otherwise
"""

import pandas as pd
import numpy as np

DATA = "data"

def load_raw():
    users = pd.read_csv(f"{DATA}/users.csv", parse_dates=["signup_date"])
    songs = pd.read_csv(f"{DATA}/songs.csv", parse_dates=["release_date"])
    history = pd.read_csv(f"{DATA}/listening_history.csv", parse_dates=["timestamp"])
    return users, songs, history


def clean(users, songs, history):
    report = {}

    # ---- duplicates ----
    report["duplicate_events_removed"] = int(history.duplicated().sum())
    history = history.drop_duplicates()

    # ---- missing values ----
    report["missing_before"] = history.isna().sum().to_dict()
    history = history.dropna(subset=["user_id", "song_id", "timestamp"])
    report["missing_after"] = history.isna().sum().to_dict()

    # ---- type sanity ----
    history["user_id"] = history["user_id"].astype(int)
    history["song_id"] = history["song_id"].astype(int)

    # ---- referential integrity: drop events referencing unknown users/songs ----
    valid_users = set(users["user_id"])
    valid_songs = set(songs["song_id"])
    before = len(history)
    history = history[history["user_id"].isin(valid_users) &
                       history["song_id"].isin(valid_songs)]
    report["orphan_events_removed"] = before - len(history)

    # ---- songs: fix impossible values ----
    songs["duration_ms"] = songs["duration_ms"].clip(lower=30_000)
    for col in ["danceability", "energy", "valence", "acousticness",
                "instrumentalness", "liveness", "speechiness"]:
        songs[col] = songs[col].clip(0, 1)
    report["songs_cleaned_rows"] = len(songs)

    return users, songs, history, report


def build_interaction_table(history):
    """
    For every (user, song) pair, compute:
      - first_listen_ts
      - total_play_count
      - repeat_within_30_days label
    """
    history = history.sort_values("timestamp")
    grp = history.groupby(["user_id", "song_id"])

    first_listen = grp["timestamp"].min().rename("first_listen_ts")
    play_count = grp.size().rename("total_play_count")

    interactions = pd.concat([first_listen, play_count], axis=1).reset_index()

    # merge back to find any listen event occurring within 30 days AFTER first listen
    merged = history.merge(interactions[["user_id", "song_id", "first_listen_ts"]],
                            on=["user_id", "song_id"])
    merged["days_since_first"] = (merged["timestamp"] - merged["first_listen_ts"]).dt.days

    repeat_flag = (
        merged[merged["days_since_first"] > 0]
        .groupby(["user_id", "song_id"])["days_since_first"]
        .apply(lambda d: int((d <= 30).any()))
        .rename("repeat_within_30_days")
    )

    interactions = interactions.merge(repeat_flag, on=["user_id", "song_id"], how="left")
    interactions["repeat_within_30_days"] = interactions["repeat_within_30_days"].fillna(0).astype(int)

    # IMPORTANT — LEAKAGE PREVENTION:
    # Any behavioral feature must be derived ONLY from the FIRST listen event,
    # never averaged across all plays (that would bake the repeat-listen outcome
    # itself into the "feature", producing an unrealistically perfect model).
    first_events = history.merge(
        interactions[["user_id", "song_id", "first_listen_ts"]],
        on=["user_id", "song_id"]
    )
    first_events = first_events[first_events["timestamp"] == first_events["first_listen_ts"]]
    first_events = first_events.drop_duplicates(subset=["user_id", "song_id"])

    first_listen_feats = first_events[[
        "user_id", "song_id", "ms_played_ratio", "skipped", "listen_context"
    ]].rename(columns={
        "ms_played_ratio": "first_listen_ms_played_ratio",
        "skipped": "first_listen_skipped",
        "listen_context": "first_listen_context",
    })

    interactions = interactions.merge(first_listen_feats, on=["user_id", "song_id"])

    # total_play_count is kept as METADATA ONLY (useful for analysis/reporting)
    # but is explicitly EXCLUDED from the model feature set in Task 4, since a
    # count > 1 almost certainly implies repeat_within_30_days == 1 (label leakage).
    return interactions


def main():
    users, songs, history = load_raw()
    users, songs, history, report = clean(users, songs, history)

    print("=== CLEANING REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    interactions = build_interaction_table(history)
    print(f"\nInteraction table shape: {interactions.shape}")
    print(f"Positive class (repeat=1) rate: {interactions['repeat_within_30_days'].mean():.3%}")

    # save cleaned artifacts
    users.to_csv(f"{DATA}/users_clean.csv", index=False)
    songs.to_csv(f"{DATA}/songs_clean.csv", index=False)
    history.to_csv(f"{DATA}/listening_history_clean.csv", index=False)
    interactions.to_csv(f"{DATA}/interactions.csv", index=False)
    print("\nSaved: users_clean.csv, songs_clean.csv, listening_history_clean.csv, interactions.csv")


if __name__ == "__main__":
    main()

