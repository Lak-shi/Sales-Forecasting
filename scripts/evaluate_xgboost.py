import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb  # Use XGBoost to load the model

# ✅ Load the trained XGBoost model correctly
model = xgb.Booster()
model.load_model("optimized_xgboost_optuna.model")  # Use XGBoost's native load function

# Load test data
df_test = pd.read_csv("xgboost_ready_sales_data.csv")

# Extract features and target
drop_columns = ["sales"]  # Always drop sales (target variable)
if "date" in df_test.columns:  # Only drop "date" if it exists
    drop_columns.append("date")

X_test = df_test.drop(columns=drop_columns)  
y_test = df_test["sales"]

# Convert to DMatrix for XGBoost prediction
X_test_dmatrix = xgb.DMatrix(X_test)

# Make predictions
y_pred = model.predict(X_test_dmatrix)

# ✅ Handle zero values in MAPE calculation
valid_idx = y_test != 0  # Ignore data points where y_test is zero
y_test_filtered = y_test[valid_idx]
y_pred_filtered = y_pred[valid_idx]

# Compute evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test_filtered - y_pred_filtered) / y_test_filtered)) * 100  # MAPE ignoring zeros
r2 = r2_score(y_test, y_pred)

print("✅ XGBoost Model Evaluation:")
print(f"📌 RMSE: {rmse:.4f}")
print(f"📌 MAPE: {mape:.4f}%")
print(f"📌 R² Score: {r2:.4f}")
