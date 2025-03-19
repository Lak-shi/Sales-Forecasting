import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load optimized model
model = load_model("lstm_sales_forecast_optimized.h5")

# Load processed data
df = pd.read_csv("processed_sales_data.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Extract sales values
data = df["sales"].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

# Prepare sequences
SEQ_LEN = 30
def create_sequences(data, seq_len):
    sequences, labels = [], []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
        labels.append(data[i + seq_len])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(data, SEQ_LEN)

# Split into Train & Test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Predict on test set
y_pred_scaled = model.predict(X_test)

# Reverse scaling
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred_scaled)

# Compute Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Results
print(f"✅ Optimized Model Evaluation:")
print(f"📌 RMSE: {rmse:.4f}")
print(f"📌 MAPE: {mape:.4%}")
print(f"📌 R² Score: {r2:.4f}")
