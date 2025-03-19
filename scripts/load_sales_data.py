import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine


# Connect to PostgreSQL inside Docker
try:
    engine = sqlalchemy.create_engine(" ")
    conn = engine.connect()
    print("✅ PostgreSQL connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")


# Load CSV file
df = pd.read_csv("data/train.csv")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Load data into PostgreSQL
df.to_sql('sales_data', engine, if_exists='replace', index=False)

print("✅ Data Loaded into PostgreSQL Successfully!")