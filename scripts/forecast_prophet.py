import pandas as pd
from prophet import Prophet
import numpy as np

# Load preprocessed data
df = pd.read_csv("processed_sales_data.csv")

# Prepare for Prophet
df = df[['date', 'sales']]
df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

# Train Prophet
prophet = Prophet()
prophet.fit(df)

# Predict future trends
future = prophet.make_future_dataframe(periods=90)
forecast = prophet.predict(future)

# Extract Prophet’s trend component
df["prophet_trend"] = forecast["trend"][:len(df)].values

# Calculate residuals (Actual - Prophet Trend)
df["residuals"] = df["y"] - df["prophet_trend"]

# Save updated dataset
df.to_csv("sales_with_prophet_trend.csv", index=False)
print("✅ Saved Prophet trend & residuals!")
