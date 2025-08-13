import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  

#Simulate 1 year of claims
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

data = []
categories = ["Car", "Home", "Health"]

for date in dates:
    num_claims = np.random.poisson(lam=5)  # average 5 claims/day
    for _ in range(num_claims):
        size = np.random.exponential(scale=2000)  # average claim ~ $2000
        category = np.random.choice(categories)
        data.append([date, size, category])

df = pd.DataFrame(data, columns=["date", "claim_size", "category"])

# ResultsSummary
print("Average claim size:", df["claim_size"].mean())
print("Total claims:", len(df))

plt.figure(figsize=(10,4))
df.groupby("date").size().plot()
plt.title("Number of Claims per Day")
plt.xlabel("Date")
plt.ylabel("Claims")
plt.show()

plt.figure(figsize=(6,4))
df["claim_size"].hist(bins=50)
plt.title("Distribution of Claim Sizes")
plt.xlabel("Claim Size ($)")
plt.ylabel("Frequency")
plt.show()

