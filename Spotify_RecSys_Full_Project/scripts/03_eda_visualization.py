"""
TASK 3 — EXPLORATORY DATA ANALYSIS & PROFESSIONAL VISUALIZATION
===================================================================
Produces 8 publication-quality charts saved to outputs/figures/
using a consistent Spotify-inspired professional color palette.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

DATA = "data"
FIG = "outputs/figures"

# ---------------------------------------------------------------
# PROFESSIONAL STYLE / PALETTE (Spotify-inspired)
# ---------------------------------------------------------------
SPOTIFY_GREEN = "#1DB954"
DARK = "#191414"
PALETTE = ["#1DB954", "#1ED760", "#191414", "#535353", "#B3B3B3",
           "#FFC864", "#E8115B", "#509BF5"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#191414",
    "text.color": "#191414",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.color": "#EDEDED",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_palette(sns.color_palette(PALETTE))


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png")


def main():
    users = pd.read_csv(f"{DATA}/users_clean.csv", parse_dates=["signup_date"])
    songs = pd.read_csv(f"{DATA}/songs_clean.csv", parse_dates=["release_date"])
    history = pd.read_csv(f"{DATA}/listening_history_clean.csv", parse_dates=["timestamp"])
    interactions = pd.read_csv(f"{DATA}/interactions.csv", parse_dates=["first_listen_ts"])

    # 1) Class balance of target
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = interactions["repeat_within_30_days"].value_counts().sort_index()
    bars = ax.bar(["No Repeat (0)", "Repeat (1)"], counts.values,
                   color=[DARK, SPOTIFY_GREEN], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:,}\n({v/counts.sum():.1%})",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Target Class Distribution\nRepeat Listen Within 30 Days")
    ax.set_ylabel("Number of (user, song) pairs")
    savefig(fig, "01_target_class_distribution")

    # 2) Genre popularity (play counts)
    genre_plays = history.merge(songs[["song_id", "genre"]], on="song_id")
    genre_counts = genre_plays["genre"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(genre_counts.index, genre_counts.values, color=SPOTIFY_GREEN)
    ax.set_title("Total Plays by Genre")
    ax.set_xlabel("Number of plays")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    savefig(fig, "02_genre_play_counts")

    # 3) Repeat rate by genre
    genre_repeat = interactions.merge(songs[["song_id", "genre"]], on="song_id")
    rate = genre_repeat.groupby("genre")["repeat_within_30_days"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [SPOTIFY_GREEN if v == rate.max() else "#535353" for v in rate.values]
    ax.barh(rate.index, rate.values * 100, color=colors)
    ax.set_title("Repeat-Listen Rate by Genre")
    ax.set_xlabel("Repeat rate (%)")
    savefig(fig, "03_repeat_rate_by_genre")

    # 4) Audio feature distributions (small multiples)
    feats = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness_db"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, f in zip(axes.flat, feats):
        sns.histplot(songs[f], bins=35, color=SPOTIFY_GREEN, edgecolor="white", ax=ax, kde=True)
        ax.set_title(f.replace("_", " ").title())
        ax.set_xlabel("")
    fig.suptitle("Distribution of Audio Features in Song Catalog", fontsize=16, fontweight="bold", y=1.02)
    savefig(fig, "04_audio_feature_distributions")

    # 5) Popularity vs repeat rate (binned)
    inter_pop = interactions.merge(songs[["song_id", "popularity"]], on="song_id")
    inter_pop["pop_bin"] = pd.cut(inter_pop["popularity"], bins=10)
    bin_rate = inter_pop.groupby("pop_bin", observed=True)["repeat_within_30_days"].mean()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in bin_rate.index]
    ax.plot(x_labels, bin_rate.values * 100, marker="o", color=SPOTIFY_GREEN,
            linewidth=2.5, markersize=7, markerfacecolor=DARK, markeredgecolor=DARK)
    ax.fill_between(range(len(x_labels)), bin_rate.values * 100, color=SPOTIFY_GREEN, alpha=0.15)
    ax.set_title("Song Popularity vs. Repeat-Listen Rate")
    ax.set_xlabel("Song popularity bucket")
    ax.set_ylabel("Repeat rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    savefig(fig, "05_popularity_vs_repeat_rate")

    # 6) Listening activity by hour-of-day and day-of-week (heatmap)
    hist_time = history.copy()
    hist_time["hour"] = hist_time["timestamp"].dt.hour
    hist_time["dow"] = hist_time["timestamp"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = hist_time.pivot_table(index="dow", columns="hour", values="song_id",
                                   aggfunc="count").reindex(dow_order)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(pivot, cmap="Greens", ax=ax, cbar_kws={"label": "Play count"}, linewidths=0.3, linecolor="white")
    ax.set_title("Listening Activity Heatmap — Hour of Day vs Day of Week")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    savefig(fig, "06_activity_heatmap")

    # 7) Subscription tier vs repeat rate
    inter_users = interactions.merge(users[["user_id", "subscription_tier"]], on="user_id")
    tier_rate = inter_users.groupby("subscription_tier")["repeat_within_30_days"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(tier_rate.index, tier_rate.values * 100, color=PALETTE[:len(tier_rate)])
    ax.set_title("Repeat-Listen Rate by Subscription Tier")
    ax.set_ylabel("Repeat rate (%)")
    savefig(fig, "07_repeat_rate_by_tier")

    # 8) Correlation heatmap of audio features
    corr = songs[feats + ["popularity"]].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Matrix — Audio Features & Popularity")
    savefig(fig, "08_feature_correlation_heatmap")

    print("\nAll EDA visualizations saved to", FIG)


if __name__ == "__main__":
    main()

