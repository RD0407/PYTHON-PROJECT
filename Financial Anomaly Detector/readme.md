EUR/CHF Anomaly Detection: Global vs Rolling vs Isolation Forest

This project compares three anomaly detection methods applied to the EUR/CHF foreign exchange rate:

Global z-score: fixed mean and standard deviation over the entire sample.

Rolling z-score: adaptive mean and standard deviation over a rolling 30-day window.

Isolation Forest (ML): unsupervised machine learning method on daily returns and volatility context.

The goal is to visualize and compare which data points are flagged as anomalies by different approaches.

## Example Output

Red circles: Global z-score anomalies

Blue crosses: Rolling z-score anomalies

Green squares: Isolation Forest anomalies
