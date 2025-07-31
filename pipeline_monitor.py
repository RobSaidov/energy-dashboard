# pipeline_monitor.py
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from config import DB_CONNECTION_STRING

def check_pipeline_health():
    """Check if the pipeline is running properly"""
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        
        # Check recent data
        query = """
        SELECT 
            COUNT(*) as record_count,
            MAX(timestamp) as latest_timestamp,
            MIN(timestamp) as earliest_timestamp
        FROM energy_consumption
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """
        
        result = pd.read_sql(query, engine)
        
        if result.empty or result['latest_timestamp'][0] is None:
            return {
                'is_healthy': False,
                'issue': 'No data found in last 24 hours',
                'last_update': None,
                'records_24h': 0,
                'expected_records_24h': 24 * 100,
                'coverage': 0
            }
        
        health_status = {
            'is_healthy': True,
            'last_update': result['latest_timestamp'][0],
            'records_24h': result['record_count'][0],
            'expected_records_24h': 24 * 100,  # 24 hours * 100 households
            'coverage': result['record_count'][0] / (24 * 100) * 100
        }
        
        # Check if data is stale (more than 90 minutes old)
        if pd.Timestamp.now() - pd.Timestamp(result['latest_timestamp'][0]) > timedelta(minutes=90):
            health_status['is_healthy'] = False
            health_status['issue'] = "Data is stale - pipeline may be down"
        
        return health_status
        
    except Exception as e:
        return {
            'is_healthy': False,
            'issue': f'Database connection error: {str(e)}',
            'last_update': None,
            'records_24h': 0,
            'expected_records_24h': 24 * 100,
            'coverage': 0
        }

if __name__ == "__main__":
    print("Checking pipeline health...")
    status = check_pipeline_health()
    
    print(f"\nPipeline Health: {'✅ Healthy' if status['is_healthy'] else '❌ Unhealthy'}")
    if not status['is_healthy']:
        print(f"Issue: {status['issue']}")
    if status['last_update']:
        print(f"Last Update: {status['last_update']}")
        print(f"24h Coverage: {status['coverage']:.1f}%")
        print(f"Records (24h): {status['records_24h']} / {status['expected_records_24h']}")