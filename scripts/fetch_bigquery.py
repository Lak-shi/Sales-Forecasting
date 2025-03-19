from google.cloud import bigquery
import pandas as pd

# Google Cloud & BigQuery Config
# PROJECT_ID = ""
# DATASET_ID = ""
# TABLE_ID = "sales_forecast"

# Initialize BigQuery Client
client = bigquery.Client()

# Query to fetch sales data
query = f"""
SELECT date, store, item, sales 
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
ORDER BY date ASC
"""

# Load data into Pandas DataFrame
df = client.query(query).to_dataframe()

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Save data to CSV
df.to_csv("sales_data_from_bigquery.csv", index=False)

print("✅ Data successfully extracted from BigQuery and saved as sales_data_from_bigquery.csv")
