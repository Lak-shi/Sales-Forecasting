import pandas as pd
from sqlalchemy import create_engine
from google.cloud import bigquery

# Step 1: Connect to PostgreSQL
engine = create_engine("")

# Step 2: Extract data from PostgreSQL
query = "SELECT * FROM sales_data;"
df = pd.read_sql(query, engine)

# Step 3: Authenticate with BigQuery using Service Account Key
client = bigquery.Client.from_service_account_json("/project/supplychain/airflow/dags/bigquery_key.json")

# Step 4: Define BigQuery Table
table_id = ""



# Step 5: Upload Data to BigQuery
job = client.load_table_from_dataframe(df, table_id)
job.result()  # Wait for job completion

print("✅ Data Uploaded to Google BigQuery Successfully!")
