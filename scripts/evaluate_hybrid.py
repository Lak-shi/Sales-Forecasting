import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Load data
df = pd.read_csv("sales_with_prophet_trend.csv")

# Load Prophet model & predict trend
prophet = Prophet()
prophet.fit(df[['ds', 'y']])
future = prophet.make_future_dataframe(periods=90)
forecast = prophet.predict(future)
df["prophet_trend"] = forecast["trend"][:len(df)].values

# Load LSTM model
model = load_model("lstm_residuals.h5")

# Prepare LSTM input (last known residuals)
SEQ_LEN = 30
residuals = df["residuals"].values.reshape(-1, 1)
X = np.array([residuals[i:i+SEQ_LEN] for i in range(len(residuals) - SEQ_LEN)])

# Predict residuals using LSTM
y_pred_residuals = model.predict(X)

# Reconstruct final sales prediction
if "hybrid_forecast" not in df.columns:
    df["hybrid_forecast"] = df["prophet_trend"].iloc[SEQ_LEN:SEQ_LEN+len(y_pred_residuals)].reset_index(drop=True) + y_pred_residuals.flatten()

# Drop NaN values to avoid errors
df["hybrid_forecast"].fillna(method="ffill", inplace=True)  # Forward fill missing values

split = int(len(df) * 0.8)
y_test = df['y'][split:].values
y_pred = df['hybrid_forecast'][split:].values


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Results
print(f"✅ Hybrid Model Evaluation:")
print(f"📌 RMSE: {rmse:.4f}")
print(f"📌 MAPE: {mape:.4%}")
print(f"📌 R² Score: {r2:.4f}")

# Save Hybrid Forecast
df.to_csv("hybrid_forecast.csv", index=False)
print("✅ Hybrid Forecast Saved!")
