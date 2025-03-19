import pandas as pd
from prophet import Prophet

# Load data
df = pd.read_csv("xgboost_ready_sales_data.csv")  # Ensure this is the correct file

# Recreate the 'date' column if it is missing
df['ds'] = pd.to_datetime(df[['year', 'month', 'day']])  # Prophet requires 'ds' column

# Prepare data for Prophet
df_prophet = df[['ds', 'sales']].rename(columns={'sales': 'y'})

# Initialize Prophet model
model = Prophet()
model.fit(df_prophet)

# Create future dataframe
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Save the trend component
df['prophet_trend'] = forecast['trend'][:len(df)]  # Align forecast with existing data

# Save processed data
df.to_csv("xgboost_with_prophet_trend.csv", index=False)
print("✅ Prophet trend extracted and saved.")
