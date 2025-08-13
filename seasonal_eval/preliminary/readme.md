# Swiss Electricity Load Forecast — Beginner Version

This project predicts Swiss electricity demand (load in MW) using simple machine learning in Python.  
It’s designed as an **easy-to-follow** example for learning time series prediction without complicated data sources or advanced ML tuning.

---

## Project Overview

We use **cleaned hourly Swiss electricity load data** from Swissgrid.  
Features used for prediction:
- Hour of the day
- Day of the week
- Month
- Load from 1 hour ago (lag_1h)
- Load from 24 hours ago (lag_24h)

We train a **Linear Regression** model to predict the next load values and compare them with actual data.

---

## Dataset

The dataset `swiss_load_hourly.csv` contains:

| datetime           | load_MW |
|--------------------|---------|
| 2025-01-01 00:00:00| 7536.42 |
| 2025-01-01 01:00:00| 8064.25 |
| 2025-01-01 02:00:00| 8008.31 |
| ...                | ...     |

---
