# EUR/CHF Anomaly Detection: Global vs Rolling vs Isolation Forest

This project compares three anomaly detection methods applied to the EUR/CHF foreign exchange rate:

Global z-score: fixed mean and standard deviation over the entire sample.

Rolling z-score: adaptive mean and standard deviation over a rolling 30-day window.

Isolation Forest (ML): unsupervised machine learning method on daily returns and volatility context.

The goal is to visualize and compare which data points are flagged as anomalies by different approaches.

## Example Output

Red circles: Global z-score anomalies

Blue crosses: Rolling z-score anomalies

Green squares: Isolation Forest anomalies

## How It Works
### Data Collection

Uses Yahoo Finance (yfinance) to download EUR/CHF daily closing prices.

Computes daily returns and rolling volatility.

### Methods

Global z-score: Flags returns exceeding ±3σ from global mean.

Rolling z-score: Flags returns exceeding ±3σ from rolling 30-day mean/std.

Isolation Forest: Uses returns + rolling volatility as features.

### Visualization

Overlays anomalies from all three methods on a single chart for comparison.

## Requirements

### Install dependencies:

```
numpy
pandas
matplotlib
yfinance
scikit-learn
```
 ## Insights

Global z-score is strict but insensitive to local volatility shifts.

Rolling z-score adapts to local regimes, flagging spikes relative to recent context.

Isolation Forest can catch nonlinear anomalies that z-scores miss.
