import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras_tuner.tuners import BayesianOptimization
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load processed data
df = pd.read_csv("processed_sales_data.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Normalize the sales data
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(df["sales"].values.reshape(-1, 1))

# Create sequences
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

# Define model building function
def build_model(hp):
    model = Sequential([
        LSTM(units=hp.Int("lstm_units", min_value=32, max_value=128, step=32), 
             return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(rate=hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)),
        LSTM(units=hp.Int("lstm_units2", min_value=32, max_value=128, step=32), return_sequences=False),
        Dropout(rate=hp.Float("dropout2", min_value=0.1, max_value=0.5, step=0.1)),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("lr", min_value=0.0001, max_value=0.01, sampling="LOG")),
        loss="mean_squared_error"
    )
    
    return model

# Bayesian Optimization Tuner
tuner = BayesianOptimization(
    build_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=2,
    directory="lstm_tuning",
    project_name="supply_chain"
)

# Run Hyperparameter Search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)

# Best Hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
print(f"✅ Best Hyperparameters:")
print(f"LSTM Units: {best_hps.get('lstm_units')}")
print(f"Dropout Rate: {best_hps.get('dropout')}")
print(f"Learning Rate: {best_hps.get('lr')}")
