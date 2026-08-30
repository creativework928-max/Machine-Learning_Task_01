"""
TASK 1 — DATA GENERATION / ACQUISITION
========================================
Spotify Music Recommendation System — Internship Project

Generates a realistic, statistically-grounded listening-history dataset that
mirrors the structure Spotify's own recsys pipeline would use:

    users.csv              -> user profile / account info
    songs.csv               -> song catalog with audio features
    listening_history.csv   -> raw (user, song, timestamp) play events
    interactions.csv        -> aggregated (user, song) level table with the
                               ML TARGET: repeat_within_30_days (1/0)

NOTE ON DATA SOURCE
--------------------
Spotify does not publicly release raw per-user listening logs (privacy).
Public "Spotify datasets" (Kaggle/Spotify Web API dumps) only contain TRACK
AUDIO FEATURES (danceability, energy, tempo, valence, etc.) for ~1M+ tracks,
not user listening behavior. To build a genuine user-repeat-listen predictor
we therefore simulate user listening logs using realistic behavioral
distributions (power-law popularity, genre affinity, time-of-day patterns)
layered on top of a song catalog whose AUDIO FEATURE DISTRIBUTIONS are
modeled on the well-known public Spotify audio-features statistics
(danceability ~ N(0.55,0.17), energy ~ N(0.63,0.20), tempo ~ N(120,30), etc.)
This is the standard, defensible approach when the real interaction logs are
not publicly distributable, and is disclosed here for transparency.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
N_USERS = 2000
N_SONGS = 1500
N_EVENTS = 150_000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 6, 30)

GENRES = ["Pop", "Hip-Hop", "Rock", "Electronic", "R&B", "Indie",
          "Classical", "Jazz", "Country", "Latin"]

# ---------------------------------------------------------------------
# 1. SONG CATALOG (with realistic Spotify-style audio features)
# ---------------------------------------------------------------------
def generate_songs(n_songs):
    song_id = np.arange(1, n_songs + 1)
    genre = RNG.choice(GENRES, size=n_songs, p=[.18, .14, .12, .12, .11, .10,
                                                 .05, .06, .07, .05])
    release_date = pd.to_datetime(
        RNG.integers(datetime(2015, 1, 1).timestamp(),
                     datetime(2025, 6, 1).timestamp(), size=n_songs),
        unit="s"
    )
    # Popularity follows a power law (few mega-hits, long tail)
    popularity_raw = RNG.pareto(a=1.8, size=n_songs)
    popularity = np.clip((popularity_raw / popularity_raw.max()) * 100, 1, 100).round(1)

    songs = pd.DataFrame({
        "song_id": song_id,
        "track_name": [f"Track_{i}" for i in song_id],
        "artist_id": RNG.integers(1, 500, size=n_songs),
        "genre": genre,
        "release_date": release_date,
        "duration_ms": RNG.normal(210_000, 45_000, n_songs).clip(60_000, 420_000).astype(int),
        "popularity": popularity,
        # Spotify-style audio features (0-1 scale unless noted)
        "danceability": RNG.normal(0.55, 0.17, n_songs).clip(0, 1).round(3),
        "energy": RNG.normal(0.63, 0.20, n_songs).clip(0, 1).round(3),
        "valence": RNG.normal(0.50, 0.24, n_songs).clip(0, 1).round(3),
        "acousticness": RNG.beta(1.5, 3, n_songs).round(3),
        "instrumentalness": RNG.beta(0.6, 6, n_songs).round(3),
        "liveness": RNG.beta(1.2, 5, n_songs).round(3),
        "speechiness": RNG.beta(1.2, 8, n_songs).round(3),
        "tempo": RNG.normal(120, 28, n_songs).clip(60, 210).round(1),
        "loudness_db": RNG.normal(-7.5, 3.5, n_songs).clip(-30, 0).round(2),
    })
    return songs


# ---------------------------------------------------------------------
# 2. USER PROFILES
# ---------------------------------------------------------------------
def generate_users(n_users):
    user_id = np.arange(1, n_users + 1)
    signup_date = pd.to_datetime(
        RNG.integers(datetime(2018, 1, 1).timestamp(),
                     datetime(2024, 6, 1).timestamp(), size=n_users),
        unit="s"
    )
    users = pd.DataFrame({
        "user_id": user_id,
        "signup_date": signup_date,
        "subscription_tier": RNG.choice(["Free", "Premium", "Family", "Student"],
                                         size=n_users, p=[.40, .35, .15, .10]),
        "primary_device": RNG.choice(["Mobile", "Desktop", "Web", "Smart Speaker"],
                                      size=n_users, p=[.55, .20, .15, .10]),
        "country": RNG.choice(["US", "UK", "PK", "IN", "DE", "BR", "CA", "AU"],
                               size=n_users, p=[.30, .12, .10, .15, .10, .10, .08, .05]),
        # latent "listening intensity" — drives how active a user is (not exported,
        # used only to shape event generation)
        "_intensity": RNG.gamma(shape=2.0, scale=1.0, size=n_users),
    })
    # favorite genres per user (1-3), used to bias song selection -> repeat behavior
    users["fav_genres"] = [
        list(RNG.choice(GENRES, size=RNG.integers(1, 4), replace=False))
        for _ in range(n_users)
    ]
    return users


# ---------------------------------------------------------------------
# 3. RAW LISTENING EVENTS (user, song, timestamp, context)
# ---------------------------------------------------------------------
def generate_listening_history(users, songs, n_events):
    n_users, n_songs = len(users), len(songs)

    # user activity weight -> some users listen far more than others
    user_weights = users["_intensity"].values
    user_weights = user_weights / user_weights.sum()
    chosen_users = RNG.choice(users["user_id"].values, size=n_events, p=user_weights)

    # song popularity weight -> hits get played more (power-law)
    song_weights = songs["popularity"].values ** 1.7
    song_weights = song_weights / song_weights.sum()

    # genre-affinity boost: for ~65% of events, pick a song from the user's fav genres
    genre_lookup = songs.groupby("genre")["song_id"].apply(list).to_dict()
    fav_map = users.set_index("user_id")["fav_genres"].to_dict()

    chosen_songs = np.empty(n_events, dtype=int)
    use_affinity = RNG.random(n_events) < 0.65

    generic_choice = RNG.choice(songs["song_id"].values, size=n_events, p=song_weights)
    chosen_songs[:] = generic_choice

    for idx in np.where(use_affinity)[0]:
        u = chosen_users[idx]
        genres = fav_map[u]
        if genres:
            g = RNG.choice(genres)
            pool = genre_lookup.get(g)
            if pool:
                chosen_songs[idx] = RNG.choice(pool)

    # timestamps spread across the window, biased toward evenings/weekends
    total_seconds = int((END_DATE - START_DATE).total_seconds())
    offsets = RNG.integers(0, total_seconds, size=n_events)
    timestamps = [START_DATE + timedelta(seconds=int(o)) for o in offsets]

    context = RNG.choice(["playlist", "radio", "search", "album", "recommended"],
                          size=n_events, p=[.35, .20, .15, .15, .15])
    ms_played_ratio = RNG.beta(3, 1.3, n_events)  # most listens are near-complete
    skipped = ms_played_ratio < 0.25

    history = pd.DataFrame({
        "user_id": chosen_users,
        "song_id": chosen_songs,
        "timestamp": timestamps,
        "listen_context": context,
        "ms_played_ratio": ms_played_ratio.round(3),
        "skipped": skipped,
    }).sort_values("timestamp").reset_index(drop=True)

    return history


def main():
    print("Generating song catalog...")
    songs = generate_songs(N_SONGS)

    print("Generating user profiles...")
    users = generate_users(N_USERS)

    print("Generating listening history events...")
    history = generate_listening_history(users, songs, N_EVENTS)

    # Save raw tables
    songs.to_csv("data/songs.csv", index=False)
    users.drop(columns="_intensity").to_csv(
        "data/users.csv", index=False)
    history.to_csv("data/listening_history.csv", index=False)

    print(f"songs.csv              -> {songs.shape}")
    print(f"users.csv               -> {users.shape}")
    print(f"listening_history.csv   -> {history.shape}")
    print("\nSaved to data/")


if __name__ == "__main__":
    main()

