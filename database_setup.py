# database_setup.py
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os

def get_db_params():
    """Get database parameters based on environment"""
    if os.getenv('DATABASE_URL'):
        # Running in Docker
        return {
            "host": "postgres",  # Docker service name
            "user": "postgres",
            "password": "312413",  # Make sure this matches docker-compose.yml
            "port": 5432
        }
    else:
        # Running locally
        return {
            "host": "localhost",
            "user": "postgres",
            "password": "312413",
            "port": 5432
        }

def create_database():
    """Create database if it doesn't exist"""
    params = get_db_params()
    
    conn = psycopg2.connect(
        host=params["host"],
        user=params["user"],
        password=params["password"],
        port=params["port"]
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'energy_dashboard'")
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute("CREATE DATABASE energy_dashboard")
        print("Database created successfully!")
    else:
        print("Database already exists!")
    
    cursor.close()
    conn.close()

def create_tables():
    """Create tables if they don't exist"""
    params = get_db_params()
    
    conn = psycopg2.connect(
        host=params["host"],
        database="energy_dashboard",
        user=params["user"],
        password=params["password"]
    )
    cursor = conn.cursor()
    
    # Create energy consumption table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS energy_consumption (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            household_id VARCHAR(50) NOT NULL,
            consumption_kwh FLOAT NOT NULL,
            temperature FLOAT,
            humidity FLOAT,
            is_weekend BOOLEAN,
            hour_of_day INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            predicted_demand FLOAT NOT NULL,
            actual_demand FLOAT,
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Tables created successfully!")

if __name__ == "__main__":
    # Set environment variable to indicate we're in Docker
    if os.path.exists('/.dockerenv') or os.getenv('DATABASE_URL'):
        os.environ['DATABASE_URL'] = 'postgresql://postgres:312413@postgres:5432/energy_dashboard'
    
    create_database()
    create_tables()