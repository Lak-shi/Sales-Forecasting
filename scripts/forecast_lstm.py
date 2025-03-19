import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load Model
model = load_model("lstm_sales_forecast.h5")

# Load processed data
df = pd.read_csv("processed_sales_data.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Prepare Last 30 Days Data for Forecasting
data = df["sales"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

last_30_days = data[-30:].reshape(1, 30, 1)

# Forecast next 90 days
future_forecast = []
for _ in range(90):
    pred = model.predict(last_30_days)
    future_forecast.append(pred[0, 0])
    last_30_days = np.roll(last_30_days, -1)
    last_30_days[0, -1, 0] = pred[0, 0]

# Reverse Scaling
future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

# Create Date Range for Predictions
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=90)

# Store Predictions
forecast_df = pd.DataFrame({"date": future_dates, "predicted_sales": future_forecast.flatten()})

# Save Predictions
forecast_df.to_csv("sales_forecast_lstm.csv", index=False)
print("✅ Sales Forecasting Complete. Results saved to sales_forecast_lstm.csv")

# Plot Forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index[-100:], df["sales"].values[-100:], label="Historical Sales")
plt.plot(future_dates, future_forecast, label="Predicted Sales", linestyle="dashed")
plt.legend()
plt.title("Sales Forecast (Next 90 Days)")
plt.show()
