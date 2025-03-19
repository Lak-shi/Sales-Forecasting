import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load preprocessed data
df = pd.read_csv("xgboost_ready_sales_data.csv")

# Define Features & Target
X = df.drop(columns=["sales"])
y = df["sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define Objective Function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    model = xgb.XGBRegressor(objective="reg:squarederror", **params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse  # Minimize RMSE

# Run Optuna Study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Get Best Hyperparameters
best_params = study.best_params
print(f"✅ Best Hyperparameters: {best_params}")

# Train with Best Hyperparameters
optimized_xgb = xgb.XGBRegressor(objective="reg:squarederror", **best_params)
optimized_xgb.fit(X_train, y_train)

# Predict & Evaluate
y_pred = optimized_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📌 Optimized RMSE: {rmse}")
print(f"📌 Optimized R² Score: {r2}")

# Save Optimized Model
optimized_xgb.save_model("optimized_xgboost_optuna.model")
print("✅ Optimized XGBoost Model Saved Successfully!")
