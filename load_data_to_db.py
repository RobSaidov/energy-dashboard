# load_data_to_db.py
import pandas as pd
from sqlalchemy import create_engine
import sys

# When running in Docker, we need to connect to 'postgres' host
# Check if we're in Docker by trying to connect to postgres first
try:
    # Try Docker connection first
    engine = create_engine('postgresql://postgres:312413@postgres:5432/energy_dashboard')
    # Test the connection
    engine.connect().close()
    print("Connected to Docker PostgreSQL")
except:
    # Fall back to localhost
    engine = create_engine('postgresql://postgres:312413@localhost/energy_dashboard')
    print("Connected to local PostgreSQL")

try:
    # Load the generated data
    df = pd.read_csv('energy_data.csv')
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Load into PostgreSQL
    df.to_sql('energy_consumption', engine, if_exists='append', index=False)
    
    print(f"Loaded {len(df)} records into the database!")
    
    # Verify the data
    result = pd.read_sql("SELECT COUNT(*) as count FROM energy_consumption", engine)
    print(f"Total records in database: {result['count'][0]}")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)