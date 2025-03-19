import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load data with Prophet trend
df = pd.read_csv("sales_with_prophet_trend.csv")

# Use residuals as LSTM target
df["residuals"] = df["residuals"].fillna(0)

# Normalize residuals
scaler = MinMaxScaler(feature_range=(-1, 1))
df["residuals"] = scaler.fit_transform(df[["residuals"]])

# Convert to sequences
SEQ_LEN = 30
def create_sequences(data, seq_len):
    sequences, labels = [], []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
        labels.append(data[i + seq_len][0])  # Predict residuals
    return np.array(sequences), np.array(labels)

features = ["residuals"]
data = df[features].values
X, y = create_sequences(data, SEQ_LEN)

# Train/Test Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(25),
    Dense(1)
])

# Compile Model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train Model
model.fit(X_train, y_train, batch_size=16, epochs=30, validation_data=(X_test, y_test))

# Save Model
model.save("lstm_residuals.h5")
print("✅ LSTM Model Trained on Residuals!")
