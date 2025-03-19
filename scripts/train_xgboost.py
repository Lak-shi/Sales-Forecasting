import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load preprocessed data
df = pd.read_csv("xgboost_ready_sales_data.csv")

# ✅ Fix: Convert `date` to datetime type and extract numerical features
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.weekday
df["week_of_year"] = df["date"].dt.isocalendar().week

# ✅ Drop the original 'date' column
df.drop(columns=["date"], inplace=True)

# Define Features & Target
X = df.drop(columns=["sales"])  # Features
y = df["sales"]  # Target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define Hyperparameter Grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Initialize Model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1
)

# Train Model
grid_search.fit(X_train, y_train)

# Get Best Model
best_model = grid_search.best_estimator_
print(f"✅ Best Hyperparameters: {grid_search.best_params_}")

# Make Predictions
y_pred = best_model.predict(X_test)

# Evaluate Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📌 RMSE: {rmse}")
print(f"📌 R² Score: {r2}")

# Save Optimized Model
best_model.save_model("optimized_xgboost.model")
print("✅ Optimized XGBoost Model Saved Successfully!")
