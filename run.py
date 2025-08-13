#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Swiss Load Forecast (token-free, using Swissgrid Excel)
------------------------------------------------------
- Reads Swissgrid Excel ("Zeitreihen0h15") with 15-min energy (kWh)
- Converts to hourly average load in MW (sum kWh per hour / 1000)
- Adds weather (Meteostat daily tavg -> hourly ffill; uses only tavg_C)
- Builds features (calendar, lags, rolling mean)
- Trains XGBoost with time-series CV
- Uses SAFE MAPE that ignores tiny actuals (<= 500 MW) to avoid divide-by-zero
- Saves plots:
    p1_entsoe_load/out/actual_vs_pred_clean.png
    p1_entsoe_load/out/actual_vs_pred_zoom.png
    p1_entsoe_load/out/error_over_time.png
    p1_entsoe_load/out/scatter_pred_vs_actual.png
    p1_entsoe_load/out/error_distribution.png
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from meteostat import Point, Daily
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# =============================
# 1) Configuration
# =============================
class Config:
    excel_path = "p2/data/swissgrid_load.xlsx"  # adjust if needed
    excel_sheet = "Zeitreihen0h15"
    outdir = "p2/out"


# =============================
# 2) Utilities
# =============================
def mape_safe(y_true, y_pred, eps=500.0) -> float:
    """
    Modified MAPE: ignore % error where actual <= eps MW (to avoid divide-by-zero).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smart_parse_decimal(series: pd.Series) -> pd.Series:
    """
    Robust number parser for '1.234,56' / '1,234.56' / '1234,56' / '1234.56'.
    Keeps dot or comma that appears LAST as the decimal.
    """
    s = (
        series.astype(str)
              .str.replace("\u202f", "", regex=False)  # narrow no-break space
              .str.replace(" ", "", regex=False)
    )

    def fix_one(x: str) -> str:
        has_dot, has_comma = "." in x, "," in x
        if has_dot and has_comma:
            if x.rfind(",") > x.rfind("."):
                return x.replace(".", "").replace(",", ".")  # 1.234,56 -> 1234.56
            else:
                return x.replace(",", "")                    # 1,234.56 -> 1234.56
        elif has_comma:
            return x.replace(",", ".")                       # 1234,56 -> 1234.56
        else:
            return x                                         # 1234.56 or digits
    s = s.map(fix_one)
    return pd.to_numeric(s, errors="coerce")


# =============================
# 3) Load Swissgrid Excel -> hourly MW
# =============================
def fetch_load_from_excel(path: str, sheet: str) -> pd.DataFrame:
    """
    Returns DataFrame indexed by hourly datetime with one column 'load_MW'.
    Source is Swissgrid Excel (15-min energy in kWh).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Excel not found: {p.resolve()}")

    xls = pd.ExcelFile(p)
    if sheet not in xls.sheet_names:
        raise ValueError(f"Sheet '{sheet}' not found. Available: {xls.sheet_names}")

    raw = pd.read_excel(xls, sheet_name=sheet)

    # First column is timestamps 'DD.MM.YYYY HH:MM' (row 0 often 'Zeitstempel')
    time_col = raw.columns[0]

    # Swiss control block total consumption column (German+English header)
    pattern = r"Summe verbrauchte Energie Regelblock Schweiz"
    candidates = [c for c in raw.columns if re.search(pattern, str(c))]
    if not candidates:
        raise ValueError("Could not find the Swiss total consumption column in the Excel.")
    cons_col = candidates[0]

    # Keep only rows that look like timestamps
    mask_ts = raw[time_col].astype(str).str.match(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}", na=False)
    df = raw.loc[mask_ts, [time_col, cons_col]].copy()

    # Parse datetime
    df["datetime"] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M", dayfirst=True)

    # Parse 15-min kWh values
    df["energy_kWh_15min"] = _smart_parse_decimal(df[cons_col])

    # Optional peek
    if len(df) > 0:
        print("First parsed 15-min kWh:", df["energy_kWh_15min"].head(4).tolist())

    # Sum 4x 15-min values per hour -> hourly kWh
    hourly_kWh = (
        df.set_index("datetime")["energy_kWh_15min"]
          .resample("1h")
          .sum()
          .dropna()
    )

    # Convert hourly kWh -> hourly MW (1 MWh/h = 1 MW)
    load_MW = (hourly_kWh / 1000.0).rename("load_MW")
    return load_MW.to_frame()


# =============================
# 4) Weather aligned to load range
# =============================
def fetch_weather_for_load(load_h: pd.DataFrame) -> pd.DataFrame:
    if load_h.empty:
        raise ValueError("Load dataframe is empty; cannot fetch aligned weather.")

    ch_center = Point(46.8, 8.23)
    start = load_h.index.min().floor("D")
    end   = load_h.index.max().ceil("D")
    wx_daily = Daily(ch_center, start, end).fetch()
    if wx_daily.empty:
        raise ValueError("Meteostat returned no weather data for the selected period.")

    if "tavg" not in wx_daily.columns:
        if {"tmin", "tmax"}.issubset(wx_daily.columns):
            wx_daily["tavg"] = (wx_daily["tmin"] + wx_daily["tmax"]) / 2.0
        else:
            wx_daily["tavg"] = 10.0

    wx = wx_daily[["tavg"]].rename(columns={"tavg": "tavg_C"})
    wx_hourly = wx.resample("1h").ffill()
    wx_hourly["tavg_C"] = (
        wx_hourly["tavg_C"]
        .ffill()
        .bfill()
        .fillna(wx_hourly["tavg_C"].mean())
    )
    return wx_hourly


# =============================
# 5) Feature engineering
# =============================
def build_features(load_h: pd.DataFrame, wx_h: pd.DataFrame) -> pd.DataFrame:
    wx_h = wx_h[["tavg_C"]].copy()
    df = load_h.join(wx_h, how="left")

    df["hour"]  = df.index.hour
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month

    for lag in [1, 24, 48, 72]:
        df[f"lag_{lag}h"] = df["load_MW"].shift(lag)

    df["roll24"] = df["load_MW"].rolling(24).mean()

    # Drop rows missing due to lags/rolling; weather already imputed
    df = df.dropna(subset=["lag_72h", "roll24"])
    return df


# =============================
# 6) Training / Evaluation / Plots
# =============================
def _mw_formatter(x, pos):
    return f"{x:,.0f}"

def _pretty_plot(idx, y_true, y_pred, title, outdir, fname):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idx, y_true, label="Actual")
    ax.plot(idx, y_pred, label="Predicted", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("MW")
    ax.yaxis.set_major_formatter(FuncFormatter(_mw_formatter))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path)
    plt.close(fig)
    print("Saved plot:", path)

def train_eval_plot(df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    n = len(df)
    if n == 0:
        raise ValueError("No rows to train on after feature engineering.")

    X = df.drop(columns=["load_MW"]).values
    y = df["load_MW"].values

    # Time-series CV folds (2..5 depending on size)
    approx_week_hours = 7 * 24
    max_splits = max(2, min(5, n // approx_week_hours))
    tscv = TimeSeriesSplit(n_splits=max_splits)

    preds, truth, idxs = [], [], []
    for tr, te in tscv.split(X):
        model = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective="reg:squarederror",
        )
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        preds.append(p); truth.append(y[te]); idxs.append(te)

    y_test = np.concatenate(truth)
    y_pred = np.concatenate(preds)
    df_test = df.iloc[np.concatenate(idxs)]

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    m = mape_safe(y_test, y_pred, eps=500.0)
    print(f"[CV {max_splits} folds] MAPE (safe): {m:.2f}%   RMSE: {rmse:.1f} MW")

    title = f"Swiss Load: Actual vs Predicted (CV) — MAPE {m:.2f}%, RMSE {rmse:.1f} MW"
    _pretty_plot(df_test.index, y_test, y_pred, title, outdir, "actual_vs_pred_clean.png")

    # Zoom: last 14 days
    if len(df_test) > 24 * 14:
        _pretty_plot(df_test.index[-24*14:], y_test[-24*14:], y_pred[-24*14:],
                     title + " (last 14 days)", outdir, "actual_vs_pred_zoom.png")

    # ---- Extra diagnostic plots ----
    # 1) Error over time
    error = y_test - y_pred
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_test.index, error)
    ax.axhline(0, color='black', lw=1)
    ax.set_title("Prediction Error Over Time (MW)")
    ax.set_ylabel("Error (Actual - Predicted)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_over_time.png"))
    plt.close(fig)

    # 2) Predicted vs Actual scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual MW"); ax.set_ylabel("Predicted MW")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_pred_vs_actual.png"))
    plt.close(fig)

    # 3) Error distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.abs(error), bins=40, edgecolor='black')
    ax.set_title("Distribution of Absolute Errors")
    ax.set_xlabel("Absolute Error (MW)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_distribution.png"))
    plt.close(fig)


# =============================
# 7) Main
# =============================
def main():
    cfg = Config()

    # 1) Load Excel -> hourly MW
    load_h = fetch_load_from_excel(cfg.excel_path, cfg.excel_sheet)

    # 2) Weather aligned to load range
    wx_h = fetch_weather_for_load(load_h)

    # Sanity prints
    print("LOAD range:", load_h.index.min(), "→", load_h.index.max(), "rows:", len(load_h))
    print("WX   range:", wx_h.index.min(),   "→", wx_h.index.max(),   "rows:", len(wx_h))

    # 3) Features
    df = build_features(load_h, wx_h)
    print("Rows after feature engineering:", len(df))

    # 4) Train / Evaluate / Plot
    train_eval_plot(df, cfg.outdir)


if __name__ == "__main__":
    main()

