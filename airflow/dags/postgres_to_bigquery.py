from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.google.cloud.transfers.postgres_to_gcs import PostgresToGCSOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from datetime import datetime

# Define DAG
dag = DAG(
    "postgres_to_bigquery",
    schedule="@daily",  # Updated syntax
    start_date=datetime(2024, 3, 1),
    catchup=False
)


# Extract PostgreSQL data & store in GCS
extract_to_gcs = PostgresToGCSOperator(
    task_id="extract_postgres_data",
    postgres_conn_id="postgres_default",
    sql="SELECT * FROM sales_data;",
    bucket="supply-chain-bucket-rosy-454122",  # Replace with your GCS bucket
    filename="sales_data.json",
    export_format="json",
    dag=dag
)

# Load GCS data into BigQuery
load_to_bigquery = GCSToBigQueryOperator(
    task_id="load_data_to_bigquery",
    bucket="supply-chain-bucket-rosy-454122",
    source_objects=["sales_data.json"],
    destination_project_dataset_table="rosy-clover-454122-b0.Supply_Chain_Optimization.sales_forecast",
    source_format="NEWLINE_DELIMITED_JSON",
    write_disposition="WRITE_TRUNCATE",
    dag=dag
)

# Define Task Dependencies
extract_to_gcs >> load_to_bigquery
