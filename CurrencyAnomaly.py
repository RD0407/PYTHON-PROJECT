#!/usr/bin/env python3

"""
Financial Time-Series Anomaly Detector — Step 1 (Simple & Robust)
- Downloads EUR/CHF daily prices (Yahoo Finance)
- Computes daily returns and z-scores
- Flags anomalies where |z| > threshold
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

SYMBOL = "EURCHF=X"
START  = "2024-01-01"
END    = "2025-08-31"
Z_THRESHOLD = 3.0
OUTDIR = Path("p4/out")


def download_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data downloaded for {symbol}. Check dates/symbol/network.")
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def compute_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    ALWAYS creates columns: Return, Zscore, Anomaly.
    If std is zero/NaN (e.g., flat series), Zscore stays NaN and Anomaly=False.
    """
    out = df.copy()
    out["Return"] = out["Close"].pct_change()

    out["Zscore"] = np.nan
    out["Anomaly"] = False

    ret = out["Return"].dropna()
    mean = ret.mean()
    std  = ret.std(ddof=1)

    if std and not np.isnan(std) and std != 0:
        z = (out["Return"] - mean) / std
        out["Zscore"] = z
        out["Anomaly"] = z.abs() > z_threshold

    return out


def plot_price_with_anomalies(df: pd.DataFrame, outdir: Path, title: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="EUR/CHF Close")

    mask = df.get("Anomaly", pd.Series(False, index=df.index)).fillna(False)
    ax.scatter(df.index[mask], df.loc[mask, "Close"], color="red", label="Anomaly")

    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel("Exchange Rate")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = outdir / "eur_chf_anomalies.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot: {out_path.resolve()}")


def main():

    prices = download_prices(SYMBOL, START, END)

    df = compute_anomalies(prices, Z_THRESHOLD)

    title = f"{SYMBOL} with Anomalies (|z| > {Z_THRESHOLD})"
    plot_price_with_anomalies(df, OUTDIR, title)

    if "Zscore" in df.columns and df["Zscore"].notna().any():
        top = (
            df.dropna(subset=["Zscore"])
              .reindex(df["Zscore"].abs().sort_values(ascending=False).index)
              .head(10)[["Close", "Return", "Zscore"]]
        )
        print("\nTop anomaly days by |z-score|:")
        print(top.to_string(float_format=lambda x: f"{x:,.4f}"))
    else:
        print("\nNo valid z-scores (series too short/flat). No anomalies to report.")

    df.to_csv(OUTDIR / "eur_chf_with_anomalies.csv", index_label="date")



if __name__ == "__main__":
    main()

