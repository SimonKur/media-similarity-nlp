import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# Load data
data_viz = pd.read_csv("fiscal_sentiment_index.csv")
data_viz["Month"] = pd.to_datetime(data_viz["Month"])

x = data_viz["Month"]
y = data_viz["FiscalSentimentIndex"]

plt.figure(figsize=(10, 5))

# Raw monthly series (context)
plt.plot(
    x, y,
    color="grey",
    linewidth=1,
    alpha=0.5,
    label="Monthly index"
)

# Linear long-run trend
z = np.polyfit(range(len(y)), y, 1)
p = np.poly1d(z)
plt.plot(
    x,
    p(range(len(y))),
    linestyle="--",
    linewidth=2,
    color="black",
    label="Linear trend"
)

# 12-month rolling mean (annual trend)
annual_trend = y.rolling(window=12, center=True).mean()
plt.plot(
    x,
    annual_trend,
    linewidth=2,
    color="#1f77b4",  # muted blue (matplotlib default)
    label="12-month rolling mean"
)

# Year-only x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=90)

# Labels & title
plt.xlabel("Year")
plt.ylabel("Fiscal Sentiment Index")
plt.title("Fiscal Sentiment Index Over Time")

# Legend
plt.legend(frameon=False)

plt.tight_layout()
plt.show()