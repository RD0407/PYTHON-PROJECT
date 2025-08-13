import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("swiss_load_hourly.csv", parse_dates=["datetime"])
df = df.set_index("datetime")

df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["lag_1h"] = df["load_MW"].shift(1)
df["lag_24h"] = df["load_MW"].shift(24)
df = df.dropna()

X = df[["hour", "dayofweek", "month", "lag_1h", "lag_24h"]]
y = df["load_MW"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.05)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

plt.figure(figsize=(12,4))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted")
plt.legend()
plt.title("Swiss Load Forecast (Simple Model)")
plt.show()

