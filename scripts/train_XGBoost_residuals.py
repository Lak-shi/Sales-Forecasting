import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (including Prophet trend)
df = pd.read_csv("xgboost_with_prophet_trend.csv")

# Convert `ds` (date) column to a numerical format (optional: extract features)
df["ds"] = pd.to_datetime(df["ds"])
df["ds_numeric"] = (df["ds"] - df["ds"].min()).dt.days  # Convert date to days since start
df.drop(columns=["ds"], inplace=True)  # Drop original date column

# Define Features & Target
X = df.drop(columns=["sales"])  # Keep `prophet_trend` as a feature
y = df["sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost with Prophet trend
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, max_depth=6)
xgb_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Hybrid Model Evaluation:")
print(f"📌 RMSE: {rmse:.4f}")
print(f"📌 R² Score: {r2:.4f}")

# Save the Hybrid Model
xgb_model.save_model("hybrid_xgboost_prophet.model")
print("✅ Hybrid XGBoost + Prophet Model Saved Successfully!")
