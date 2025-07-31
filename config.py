# config.py
import os

IS_DOCKER = os.getenv('DATABASE_URL') is not None

if IS_DOCKER:
    DB_CONNECTION_STRING = os.getenv('DATABASE_URL')

else:
    # Database configuration
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'energy_dashboard',
        'user': 'postgres',
        'password': '312413', 
        'port': 5432
    }

    # Create connection string
    DB_CONNECTION_STRING = "postgresql://postgres:312413@localhost:5432/energy_dashboard"

# Weather API (optional)
WEATHER_API_KEY = "c65065f3d7e199c85bbf4fa717cdf77e"

# Pipeline settings
PIPELINE_INTERVAL_MINUTES = 60
HOUSEHOLDS_COUNT = 100
DATA_RETENTION_DAYS = 30