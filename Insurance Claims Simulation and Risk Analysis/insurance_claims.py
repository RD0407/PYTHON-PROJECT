#!/usr/bin/env python3

"""
Insurance Claims — Frequency/Severity Modeling + Portfolio Risk
---------------------------------------------------------------
Here I:
1) simulate a year of claims (dates, sizes, categories)
2) model frequency (how many claims happen) with a Poisson GLM (+ weekday effect)
3) model severity (how big claims are) with Gamma and LogNormal, compare fits
4) simulate total losses for many "years" (Monte Carlo) to estimate risk
5) save plots + simple risk metrics (mean, 95% VaR/TVaR)

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy import stats
import statsmodels.api as sm

# -----------------------------
class CFG:
    seed = 42
    start = "2024-01-01"
    end = "2024-12-31"
    mean_claims_per_day = 5.0     # average daily frequency
    sev_scale = 2000.0            # ~mean severity for exponential-ish base
    outdir = Path("p3/out")
    n_years_sim = 10000           # Monte Carlo years for portfolio risk


# 1) Simulate one year of claims 
# ******************************************************
def simulate_claims(cfg: CFG) -> pd.DataFrame:
    """
    We make a toy dataset:
      - For each day, number of claims ~ Poisson(lambda_day)
      - Each claim has a size (we'll mix two exponentials to create a fat tail)
      - Category is random
      - Add a weekday pattern (Mon-Fri slightly higher)
    """
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start, cfg.end, freq="D")

    # Weekday uplift: M-F  +20%, Sat  -10%, Sun  -15% (toy pattern)
    weekday_uplift = {0: 1.20, 1: 1.20, 2: 1.20, 3: 1.20, 4: 1.20, 5: 0.90, 6: 0.85}

    categories = ["Car", "Home", "Health"]
    rows = []

    for d in dates:
        lam = cfg.mean_claims_per_day * weekday_uplift[d.weekday()]
        n = rng.poisson(lam=lam)  # number of claims today

        for _ in range(n):
            # severity: mixture to get a long right tail
            if rng.random() < 0.85:
                sev = rng.exponential(scale=cfg.sev_scale)         
            else:
                sev = rng.exponential(scale=cfg.sev_scale * 8.0) 

            cat = rng.choice(categories)
            rows.append((d, float(sev), cat))

    df = pd.DataFrame(rows, columns=["date", "claim_size", "category"])
    return df

# 2) Frequency model (Poisson GLM with weekday)
# *****************************************************************************************
def fit_frequency_glm(df_claims: pd.DataFrame) -> dict:
    """
    Build daily counts and fit a Poisson GLM:
        count ~ weekday (categorical, Monday baseline)
    Returns fitted model, daily DataFrame, and design matrix X.
    """
    # Daily counts, ensure a full daily index (fill any gaps with 0)
    daily = (
        df_claims.groupby("date")
        .size()
        .rename("count")
        .to_frame()
        .asfreq("D", fill_value=0)
    )

    # Day of week (0=Mon..6=Sun)
    daily["dow"] = daily.index.dayofweek

    # One-hot encode weekday with Monday as baseline (drop_first=True)
    dummies = pd.get_dummies(daily["dow"], prefix="dow", drop_first=True)

    # Ensure ALL expected dummy columns exist (Tue..Sun = 1..6)
    expected = [f"dow_{i}" for i in range(1, 7)]
    for col in expected:
        if col not in dummies.columns:
            dummies[col] = 0

    # Order columns consistently and cast to float
    dummies = dummies[expected].astype(float)

    # Add intercept/constant
    X = sm.add_constant(dummies, has_constant="add")

    # Target (counts) as float (GLM will handle it, though they're counts)
    y = daily["count"].astype(float).values

    # Final sanity: replace any NaN/inf that might sneak in (shouldn't now)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Fit Poisson GLM with log link
    model = sm.GLM(y, X.values, family=sm.families.Poisson())
    result = model.fit()

    # Predicted mean counts per day (lambda_hat)
    daily["lambda_hat"] = result.predict(X.values)

    return {"result": result, "daily": daily, "X": X}


# 3) Severity modeling: fit Gamma and LogNormal, compare
# ***************************************************************
def fit_severity_models(df_claims: pd.DataFrame) -> dict:
    """
    Fit Gamma and LogNormal to claim_size (MLE via scipy.stats.fit).
    Compare log-likelihood (and AIC). Return best model + params.
    """
    y = df_claims["claim_size"].values
    y = y[y > 0]  # ensure positivity

    # Fit Gamma: scipy returns (shape, loc, scale)
    gamma_params = stats.gamma.fit(y, floc=0)  # force loc=0 for stability
    # Fit LogNormal: scipy uses shape=sigma, loc, scale=exp(mu)
    lnorm_params = stats.lognorm.fit(y, floc=0)

    # Log-likelihoods
    ll_gamma = np.sum(stats.gamma.logpdf(y, *gamma_params))
    ll_lnorm = np.sum(stats.lognorm.logpdf(y, *lnorm_params))

    # AIC = 2k - 2LL  (k = number of parameters)
    k_gamma = 2  # (shape, scale) with loc fixed=0
    k_lnorm = 2  # (shape, scale) with loc fixed=0
    aic_gamma = 2 * k_gamma - 2 * ll_gamma
    aic_lnorm = 2 * k_lnorm - 2 * ll_lnorm

    best_name = "gamma" if aic_gamma < aic_lnorm else "lognorm"
    best_params = gamma_params if best_name == "gamma" else lnorm_params

    return {
        "gamma": {"params": gamma_params, "ll": ll_gamma, "aic": aic_gamma},
        "lognorm": {"params": lnorm_params, "ll": ll_lnorm, "aic": aic_lnorm},
        "best": {"name": best_name, "params": best_params}
    }

# 4) Monte Carlo simulation of aggregate losses
#************************************************************
def simulate_aggregate_losses(freq_fit: dict, sev_fit: dict, cfg: CFG) -> pd.DataFrame:
    """
    Simulate n_years of total ANNUAL losses:
      - For each simulated "year": loop through 365 days
      - For each day: draw claim count ~ Poisson(lambda_hat by weekday)
      - For each claim: draw severity ~ best-fit distribution
      - Sum to annual total
    Returns DataFrame of annual totals.
    """
    rng = np.random.default_rng(cfg.seed + 99)

    # Pull weekday effects from the fitted GLM (average lambda by weekday):
    daily = freq_fit["daily"].copy()
    # compute mean lambda by dow (0..6)
    lam_by_dow = daily.groupby(daily.index.dayofweek)["lambda_hat"].mean().reindex(range(7)).values

    # Severity sampler
    best = sev_fit["best"]["name"]
    p = sev_fit["best"]["params"]

    def sample_severity(n):
        if best == "gamma":
            shape, loc, scale = p
            return stats.gamma.rvs(shape, loc=loc, scale=scale, size=n, random_state=rng)
        else:
            shape, loc, scale = p
            return stats.lognorm.rvs(shape, loc=loc, scale=scale, size=n, random_state=rng)

    # Simulate
    totals = []
    for _ in range(cfg.n_years_sim):
        total = 0.0
        # simulate a year with weekday pattern: start on Monday for simplicity
        # (the exact alignment barely matters for large n_years)
        for day_idx in range(365):
            dow = day_idx % 7
            lam = lam_by_dow[dow]
            nclaims = rng.poisson(lam=lam)
            if nclaims > 0:
                total += float(np.sum(sample_severity(nclaims)))
        totals.append(total)

    return pd.DataFrame({"annual_total_loss": totals})


# -----------------------------
# 5) Plots
# -----------------------------
def plot_frequency(daily: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Counts & fitted mean
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily["count"], label="Observed daily count")
    ax.plot(daily.index, daily["lambda_hat"], label="Fitted mean (Poisson GLM)")
    ax.set_title("Daily Claim Frequency: Observed vs Fitted Mean")
    ax.set_xlabel("Date"); ax.set_ylabel("Claims per day")
    ax.legend(); fig.tight_layout()
    fig.savefig(outdir / "frequency_observed_vs_fitted.png"); plt.close(fig)


def plot_severity_fit(df_claims: pd.DataFrame, sev_fit: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    y = df_claims["claim_size"].values
    y = y[y > 0]

    # Histogram + fitted PDFs
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=80, density=True, alpha=0.5, edgecolor="black")
    xs = np.linspace(0, np.percentile(y, 99.5), 500)

    g_shape, g_loc, g_scale = sev_fit["gamma"]["params"]
    ln_shape, ln_loc, ln_scale = sev_fit["lognorm"]["params"]

    ax.plot(xs, stats.gamma.pdf(xs, g_shape, loc=g_loc, scale=g_scale), label=f"Gamma (AIC={sev_fit['gamma']['aic']:.1f})")
    ax.plot(xs, stats.lognorm.pdf(xs, ln_shape, loc=ln_loc, scale=ln_scale), label=f"LogNormal (AIC={sev_fit['lognorm']['aic']:.1f})")

    ax.set_title("Claim Severity: Histogram with Fitted Distributions")
    ax.set_xlabel("Claim Size"); ax.set_ylabel("Density"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "severity_fits.png"); plt.close(fig)


def plot_annual_aggregate(df_agg: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Histogram of annual totals
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_agg["annual_total_loss"].values, bins=50, edgecolor="black")
    ax.set_title("Simulated Annual Aggregate Losses")
    ax.set_xlabel("Annual Total Loss"); ax.set_ylabel("Frequency")
    fig.tight_layout(); fig.savefig(outdir / "annual_aggregate_hist.png"); plt.close(fig)


# -----------------------------
# 6) Utility: simple risk metrics
# -----------------------------
def risk_metrics(df_agg: pd.DataFrame) -> dict:
    vals = np.sort(df_agg["annual_total_loss"].values)
    mean = float(np.mean(vals))
    p95 = float(np.quantile(vals, 0.95))
    # TVaR(95%) = mean of the worst 5%
    tail = vals[int(0.95 * len(vals)):]
    tvar95 = float(np.mean(tail)) if len(tail) > 0 else float("nan")
    return {"mean": mean, "VaR95": p95, "TVaR95": tvar95}


# -----------------------------
# 7) Main flow
# -----------------------------
def main():
    cfg = CFG()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    # A) Simulate base data
    df = simulate_claims(cfg)

    # Quick summaries
    print("=== Simulated Data ===")
    print("Rows (claims):", len(df))
    print("Avg claim size:", df["claim_size"].mean())
    print("Claims/day (min/mean/max):",
          df.groupby("date").size().min(),
          df.groupby("date").size().mean(),
          df.groupby("date").size().max())

    # B) Fit frequency model (Poisson GLM with weekday)
    freq_fit = fit_frequency_glm(df)
    print("\n=== Frequency Model (Poisson GLM) Summary ===")
    print(freq_fit["result"].summary())

    # C) Fit severity models (Gamma vs LogNormal)
    sev_fit = fit_severity_models(df)
    print("\n=== Severity Fit Comparison ===")
    print(f"Gamma   AIC: {sev_fit['gamma']['aic']:.2f}")
    print(f"LogNorm AIC: {sev_fit['lognorm']['aic']:.2f}")
    print(f"Best model: {sev_fit['best']['name']}  params={sev_fit['best']['params']}")

    # D) Simulate annual aggregates (portfolio risk)
    df_agg = simulate_aggregate_losses(freq_fit, sev_fit, cfg)
    df_agg.to_csv(cfg.outdir / "annual_aggregate_sim.csv", index=False)

    # E) Risk metrics
    rm = risk_metrics(df_agg)
    print("\n=== Risk Metrics (Annual) ===")
    print(f"Mean:   {rm['mean']:.0f}")
    print(f"VaR95:  {rm['VaR95']:.0f}")
    print(f"TVaR95: {rm['TVaR95']:.0f}")


    plot_frequency(freq_fit["daily"], cfg.outdir)
    plot_severity_fit(df, sev_fit, cfg.outdir)
    plot_annual_aggregate(df_agg, cfg.outdir)

    print(f"\nSaved outputs to: {cfg.outdir.resolve()}")
    print(" - frequency_observed_vs_fitted.png")
    print(" - severity_fits.png")
    print(" - annual_aggregate_hist.png")
    print(" - annual_aggregate_sim.csv")


if __name__ == "__main__":
    main()

