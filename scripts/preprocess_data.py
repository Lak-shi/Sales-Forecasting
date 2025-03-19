# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # Load data
# df = pd.read_csv("sales_data_from_bigquery.csv")

# # Convert date column to datetime
# df["date"] = pd.to_datetime(df["date"])

# # Create additional time-based features
# df["day_of_week"] = df["date"].dt.dayofweek
# df["month"] = df["date"].dt.month
# df["year"] = df["date"].dt.year
# df["lag_7"] = df["sales"].shift(7)  # Sales from last week
# df["lag_30"] = df["sales"].shift(30)  # Sales from last month
# df["rolling_7"] = df["sales"].rolling(window=7).mean()  # 7-day average
# df["rolling_30"] = df["sales"].rolling(window=30).mean()  # 30-day average

# # Drop NaN values created by shifting
# df.dropna(inplace=True)

# # Normalize sales data
# scaler = MinMaxScaler(feature_range=(0,1))
# df["sales"] = scaler.fit_transform(df[["sales"]])

# # Save processed data
# df.to_csv("processed_sales_data_with_features.csv", index=False)
# print("✅ Data preprocessing with additional features complete.")

# import pandas as pd
# import numpy as np

# # Load original dataset (ensure it has correct structure)
# df = pd.read_csv("/Users/lakshitashetty/Desktop/project/supplychain/data/sales_data.csv")
# print("✅ Loaded sales_data.csv - Columns:", df.columns)

# # Ensure date column is datetime type
# df['date'] = pd.to_datetime(df['date'])

# # Sort data by store, item, date
# df = df.sort_values(by=['store', 'item', 'date'])

# # Feature Engineering: Create lag features
# LAGS = [1, 7, 14, 30]  
# for lag in LAGS:
#     df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)

# # Rolling Window Features
# df['sales_moving_avg_7'] = df.groupby(['store', 'item'])['sales'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# # Drop NaN values caused by shifting
# df.dropna(inplace=True)

# # Save preprocessed data
# df.to_csv("xgboost_ready_sales_data.csv", index=False)
# print("✅ Data preprocessing for XGBoost complete. Saved as xgboost_ready_sales_data.csv")


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("xgboost_ready_sales_data.csv")

# Ensure date column is datetime type
df["date"] = pd.to_datetime(df["date"])

# Convert date into useful numerical features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["weekofyear"] = df["date"].dt.isocalendar().week

# Drop original 'date' column (as XGBoost does not accept object columns)
df.drop(columns=["date"], inplace=True)

# Save the preprocessed data
df.to_csv("xgboost_ready_sales_data.csv", index=False)

print("✅ Data preprocessing updated: 'date' column converted into numerical features!")
