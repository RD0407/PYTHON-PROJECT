#!/usr/bin/env python3
"""
Comparison of Anomaly Detection Methods on EUR/CHF
--------------------------------------------------
- Global z-score (fixed mean/std over whole period)
- Rolling z-score (adaptive mean/std over last 30 days)
- Isolation Forest (unsupervised ML on returns + rolling std)

Saves plot: p4/out/comparison_anomalies.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import IsolationForest

# Config
SYMBOL = "EURCHF=X"
START, END = "2025-01-01", "2025-08-01"
OUTDIR = Path("p4/out")
Z_THRESHOLD = 3.0
ROLLING_WINDOW = 30
IF_CONTAM = 0.02


# -----------------
# 1) Data download
# -----------------
def download_prices(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    df["Return"] = df["Close"].pct_change()
    return df


# -----------------
# 2) Global z-score
# -----------------
def global_zscore(df, threshold=3.0):
    df = df.copy()
    ret = df["Return"].dropna()
    mu, sigma = ret.mean(), ret.std(ddof=1)
    df["Z_global"] = (df["Return"] - mu) / sigma
    df["Anom_global"] = df["Z_global"].abs() > threshold
    return df


# -----------------
# 3) Rolling z-score
# -----------------
def rolling_zscore(df, window=30, threshold=3.0):
    df = df.copy()
    roll_mean = df["Return"].rolling(window, min_periods=5).mean()
    roll_std = df["Return"].rolling(window, min_periods=5).std(ddof=1)
    df["Z_roll"] = (df["Return"] - roll_mean) / roll_std
    df["Anom_roll"] = df["Z_roll"].abs() > threshold
    return df


# -----------------
# 4) Isolation Forest
# -----------------
def isolation_forest(df, contamination=0.02):
    df = df.copy()
    # Features: return + rolling std (to give context of volatility)
    roll_std = df["Return"].rolling(ROLLING_WINDOW, min_periods=5).std(ddof=1)
    features = pd.DataFrame({
        "return": df["Return"].fillna(0),
        "roll_std": roll_std.fillna(0)
    })
    clf = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42
    )
    labels = clf.fit_predict(features)
    df["Anom_if"] = labels == -1
    return df


# -----------------
# 5) Plotting
# -----------------
def plot_comparison(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Base price line
    ax.plot(df.index, df["Close"], label="EUR/CHF Close", color="black")

    # Mark anomalies from each method
    ax.scatter(df.index[df["Anom_global"]],
               df["Close"][df["Anom_global"]],
               color="red", marker="o", label="Global z-score")
    ax.scatter(df.index[df["Anom_roll"]],
               df["Close"][df["Anom_roll"]],
               color="blue", marker="x", label="Rolling z-score")
    ax.scatter(df.index[df["Anom_if"]],
               df["Close"][df["Anom_if"]],
               color="green", marker="s", label="Isolation Forest")

    ax.set_title("EUR/CHF Anomaly Detection: Global vs Rolling vs Isolation Forest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exchange Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = outdir / "comparison_anomalies.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot: {out_path.resolve()}")


# -----------------
# 6) Main
# -----------------
def main():
    df = download_prices(SYMBOL, START, END)
    df = global_zscore(df, Z_THRESHOLD)
    df = rolling_zscore(df, ROLLING_WINDOW, Z_THRESHOLD)
    df = isolation_forest(df, IF_CONTAM)

    plot_comparison(df, OUTDIR)


if __name__ == "__main__":
    main()

