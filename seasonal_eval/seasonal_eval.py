#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seasonal evaluation of Swiss load model (Winter vs Summer)
- Re-reads Swissgrid Excel, converts to hourly MW
- Adds Meteostat weather, builds features
- Time-series CV → out-of-fold predictions
- Computes seasonal metrics (safe MAPE, RMSE) for:
    • Winter (Dec–Feb)
    • Summer (Jun–Aug)
- Saves:
    p1_entsoe_load/out/seasonal_metrics.txt
    p1_entsoe_load/out/winter_scatter.png
    p1_entsoe_load/out/summer_scatter.png
    p1_entsoe_load/out/winter_error_hist.png
    p1_entsoe_load/out/summer_error_hist.png
    p1_entsoe_load/out/winter_profile_pred_vs_actual.png
    p1_entsoe_load/out/summer_profile_pred_vs_actual.png
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


# -----------------------
# Config (adjust if needed)
# -----------------------
EXCEL_PATH = "p2/data/swissgrid_load.xlsx"
EXCEL_SHEET = "Zeitreihen0h15"
OUTDIR = "p2/out"


# -----------------------
# Helpers
# -----------------------
def mape_safe(y_true, y_pred, eps=500.0) -> float:
    """Modified MAPE: ignore % error when actual <= eps MW."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smart_parse_decimal(series: pd.Series) -> pd.Series:
    """Parse '1.234,56' / '1,234.56' / '1234,56' / '1234.56' robustly."""
    s = (
        series.astype(str)
              .str.replace("\u202f", "", regex=False)
              .str.replace(" ", "", regex=False)
    )

    def fix_one(x: str) -> str:
        has_dot, has_comma = "." in x, "," in x
        if has_dot and has_comma:
            if x.rfind(",") > x.rfind("."):
                return x.replace(".", "").replace(",", ".")
            else:
                return x.replace(",", "")
        elif has_comma:
            return x.replace(",", ".")
        else:
            return x
    s = s.map(fix_one)
    return pd.to_numeric(s, errors="coerce")


def fetch_load_from_excel(path: str, sheet: str) -> pd.DataFrame:
    """Swissgrid Excel (15-min kWh) → hourly MW DataFrame indexed by datetime."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Excel not found: {p.resolve()}")
    xls = pd.ExcelFile(p)
    if sheet not in xls.sheet_names:
        raise ValueError(f"Sheet '{sheet}' not found. Available: {xls.sheet_names}")

    raw = pd.read_excel(xls, sheet_name=sheet)
    time_col = raw.columns[0]

    pattern = r"Summe verbrauchte Energie Regelblock Schweiz"
    candidates = [c for c in raw.columns if re.search(pattern, str(c))]
    if not candidates:
        raise ValueError("Swiss total consumption column not found.")
    cons_col = candidates[0]

    mask_ts = raw[time_col].astype(str).str.match(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}", na=False)
    df = raw.loc[mask_ts, [time_col, cons_col]].copy()
    df["datetime"] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M", dayfirst=True)
    df["energy_kWh_15min"] = _smart_parse_decimal(df[cons_col])

    hourly_kWh = df.set_index("datetime")["energy_kWh_15min"].resample("1h").sum().dropna()
    load_MW = (hourly_kWh / 1000.0).rename("load_MW")
    return load_MW.to_frame()


def fetch_weather_for_load(load_h: pd.DataFrame) -> pd.DataFrame:
    """Meteostat daily tavg → hourly forward-filled; aligned to load horizon."""
    if load_h.empty:
        raise ValueError("Empty load dataframe.")
    ch_center = Point(46.8, 8.23)
    start = load_h.index.min().floor("D")
    end = load_h.index.max().ceil("D")
    wx_daily = Daily(ch_center, start, end).fetch()
    if wx_daily.empty:
        raise ValueError("No Meteostat data for the selected period.")
    if "tavg" not in wx_daily.columns:
        if {"tmin", "tmax"}.issubset(wx_daily.columns):
            wx_daily["tavg"] = (wx_daily["tmin"] + wx_daily["tmax"]) / 2.0
        else:
            wx_daily["tavg"] = 10.0
    wx = wx_daily[["tavg"]].rename(columns={"tavg": "tavg_C"})
    wx_hourly = wx.resample("1h").ffill()
    wx_hourly["tavg_C"] = wx_hourly["tavg_C"].ffill().bfill().fillna(wx_hourly["tavg_C"].mean())
    return wx_hourly


def build_features(load_h: pd.DataFrame, wx_h: pd.DataFrame) -> pd.DataFrame:
    """Join load+weather, add calendar, lags, rolling mean; drop rows missing due to lags."""
    df = load_h.join(wx_h[["tavg_C"]], how="left")
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    for lag in [1, 24, 48, 72]:
        df[f"lag_{lag}h"] = df["load_MW"].shift(lag)
    df["roll24"] = df["load_MW"].rolling(24).mean()
    df = df.dropna(subset=["lag_72h", "roll24"])
    return df


# -----------------------
# Training + seasonal evaluation
# -----------------------
def seasonal_eval_and_plots(df: pd.DataFrame, outdir: str):
    """Time-series CV → OOF predictions; seasonal metrics + plots."""
    os.makedirs(outdir, exist_ok=True)
    X = df.drop(columns=["load_MW"]).values
    y = df["load_MW"].values

    # Build OOF predictions
    n = len(df)
    approx_week_hours = 7 * 24
    n_splits = max(2, min(5, n // approx_week_hours))
    tscv = TimeSeriesSplit(n_splits=n_splits)
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
    idx = df.index[np.concatenate(idxs)]

    # DataFrame with results
    res = pd.DataFrame({"datetime": idx, "actual": y_test, "pred": y_pred})
    res["month"] = res["datetime"].dt.month
    res["hour"] = res["datetime"].dt.hour
    res = res.sort_values("datetime")

    # Seasonal masks
    winter = res["month"].isin([12, 1, 2])
    summer = res["month"].isin([6, 7, 8])

    # Metrics
    def metrics(mask, name):
        yy = res.loc[mask, "actual"].values
        pp = res.loc[mask, "pred"].values
        rmse = float(np.sqrt(mean_squared_error(yy, pp)))
        mape = mape_safe(yy, pp, eps=500.0)
        return f"{name}: MAPE {mape:.2f}%, RMSE {rmse:.1f} MW\n"

    with open(os.path.join(outdir, "seasonal_metrics.txt"), "w") as f:
        f.write(metrics(winter, "Winter (Dec–Feb)"))
        f.write(metrics(summer, "Summer (Jun–Aug)"))

    # ---- Plots ----
    def fmt_mw(x, pos): return f"{x:,.0f}"
    yfmt = FuncFormatter(fmt_mw)

    # Scatter (winter)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(res.loc[winter, "actual"], res.loc[winter, "pred"], alpha=0.4)
    lims = [min(res.loc[winter, "actual"].min(), res.loc[winter, "pred"].min()),
            max(res.loc[winter, "actual"].max(), res.loc[winter, "pred"].max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual MW"); ax.set_ylabel("Predicted MW")
    ax.set_title("Predicted vs Actual — Winter")
    ax.yaxis.set_major_formatter(yfmt)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "winter_scatter.png")); plt.close(fig)

    # Scatter (summer)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(res.loc[summer, "actual"], res.loc[summer, "pred"], alpha=0.4)
    lims = [min(res.loc[summer, "actual"].min(), res.loc[summer, "pred"].min()),
            max(res.loc[summer, "actual"].max(), res.loc[summer, "pred"].max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual MW"); ax.set_ylabel("Predicted MW")
    ax.set_title("Predicted vs Actual — Summer")
    ax.yaxis.set_major_formatter(yfmt)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "summer_scatter.png")); plt.close(fig)

    # Error distribution (winter)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.abs(res.loc[winter, "actual"] - res.loc[winter, "pred"]), bins=40, edgecolor="black")
    ax.set_title("Absolute Error Distribution — Winter")
    ax.set_xlabel("Absolute Error (MW)"); ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(yfmt)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "winter_error_hist.png")); plt.close(fig)

    # Error distribution (summer)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.abs(res.loc[summer, "actual"] - res.loc[summer, "pred"]), bins=40, edgecolor="black")
    ax.set_title("Absolute Error Distribution — Summer")
    ax.set_xlabel("Absolute Error (MW)"); ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(yfmt)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "summer_error_hist.png")); plt.close(fig)

    # Average daily profile of Actual vs Predicted — Winter
    winter_prof = res.loc[winter].groupby("hour")[["actual", "pred"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(winter_prof.index, winter_prof["actual"], label="Actual", linewidth=2)
    ax.plot(winter_prof.index, winter_prof["pred"],   label="Predicted", linewidth=2)
    ax.set_title("Average Daily Load Profile — Winter (Dec–Feb)")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("MW"); ax.legend(); ax.grid(True)
    ax.yaxis.set_major_formatter(yfmt)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "winter_profile_pred_vs_actual.png")); plt.close(fig)

    # Average daily profile of Actual vs Predicted — Summer
    summer_prof = res.loc[summer].groupby("hour")[["actual", "pred"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summer_prof.index, summer_prof["actual"], label="Actual", linewidth=2)
    ax.plot(summer_prof.index, summer_prof["pred"],   label="Predicted", linewidth=2)
    ax.set_title("Average Daily Load Profile — Summer (Jun–Aug)")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("MW"); ax.legend(); ax.grid(True)
    ax.yaxis.set_major_formatter(yfmt)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "summer_profile_pred_vs_actual.png")); plt.close(fig)

    print("✅ Wrote seasonal metrics & plots to:", Path(outdir).resolve())


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    load_h = fetch_load_from_excel(EXCEL_PATH, EXCEL_SHEET)
    wx_h   = fetch_weather_for_load(load_h)
    df     = build_features(load_h, wx_h)

    print("Data rows after features:", len(df))
    seasonal_eval_and_plots(df, OUTDIR)


if __name__ == "__main__":
    main()

