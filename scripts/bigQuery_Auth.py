from google.cloud import bigquery

# Authenticate using Service Account Key
client = bigquery.Client.from_service_account_json("project/supplychain/airflow/dags/bigquery_key.json")

# Test query
query = "SELECT CURRENT_DATE() AS today"
result = client.query(query).to_dataframe()

print("✅ BigQuery Authentication Successful!")
print(result)
