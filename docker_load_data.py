# docker_load_data.py
import pandas as pd
from sqlalchemy import create_engine

# Force Docker connection
engine = create_engine('postgresql://postgres:312413@postgres:5432/energy_dashboard')

# Load and insert data
df = pd.read_csv('energy_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.to_sql('energy_consumption', engine, if_exists='append', index=False)

print(f"Loaded {len(df)} records!")