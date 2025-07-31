# data_pipeline.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import requests
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import json
import os
from typing import Dict, List
import joblib
from config import DB_CONNECTION_STRING, WEATHER_API_KEY, PIPELINE_INTERVAL_MINUTES, HOUSEHOLDS_COUNT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnergyDataPipeline:
    def __init__(self):
        """Initialize the data pipeline"""
        self.engine = create_engine(DB_CONNECTION_STRING)
        self.weather_api_key = WEATHER_API_KEY
        self.households = [f'HH_{i:03d}' for i in range(HOUSEHOLDS_COUNT)]
        self.model = None
        
        # Load ML model for predictions
        try:
            self.model = joblib.load('energy_demand_model.pkl')
            logger.info("ML model loaded successfully")
        except:
            logger.warning("ML model not found. Predictions will be skipped.")
    
    def fetch_weather_data(self, city: str = "San Francisco") -> Dict:
        """Fetch current weather data from OpenWeatherMap API"""
        if not self.weather_api_key:
            # Return simulated weather data if no API key
            hour = datetime.now().hour
            temp = 20 + 10 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 2)
            humidity = 60 + 20 * np.sin(hour * np.pi / 24) + np.random.normal(0, 5)
            return {
                'temperature': round(temp, 1),
                'humidity': round(humidity, 1),
                'weather': 'simulated'
            }
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            return {
                'temperature': round(data['main']['temp'], 1),
                'humidity': round(data['main']['humidity'], 1),
                'weather': data['weather'][0]['main']
            }
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            # Fallback to simulated data
            return self.fetch_weather_data(None)
    
    def generate_consumption_data(self, weather: Dict) -> pd.DataFrame:
        """Generate realistic consumption data based on current conditions"""
        timestamp = datetime.now()
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        # Base consumption patterns
        if 6 <= hour <= 9 or 17 <= hour <= 22:  # Peak hours
            base_consumption = np.random.normal(2.5, 0.5)
        elif 23 <= hour or hour <= 5:  # Night hours
            base_consumption = np.random.normal(0.8, 0.2)
        else:  # Day hours
            base_consumption = np.random.normal(1.5, 0.3)
        
        # Weather impact
        temp = weather['temperature']
        if temp > 28:  # Hot - AC usage
            base_consumption *= 1.3 + (temp - 28) * 0.05
        elif temp < 10:  # Cold - Heating usage
            base_consumption *= 1.3 + (10 - temp) * 0.05
        
        # Weekend adjustment
        if is_weekend:
            base_consumption *= 1.1
        
        # Generate data for each household
        data = []
        for household in self.households:
            # Add household-specific variation
            household_factor = 0.8 + 0.4 * hash(household) % 10 / 10
            consumption = max(0.1, base_consumption * household_factor + np.random.normal(0, 0.2))
            
            # Occasionally add anomalies (1% chance)
            if np.random.random() < 0.01:
                consumption *= np.random.uniform(2, 4)  # Spike
                logger.info(f"Anomaly generated for {household}: {consumption:.2f} kWh")
            
            data.append({
                'timestamp': timestamp,
                'household_id': household,
                'consumption_kwh': round(consumption, 3),
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'is_weekend': is_weekend,
                'hour_of_day': hour
            })
        
        return pd.DataFrame(data)
    
    def save_to_database(self, df: pd.DataFrame) -> None:
        """Save data to PostgreSQL database"""
        try:
            df.to_sql('energy_consumption', self.engine, if_exists='append', index=False)
            logger.info(f"Saved {len(df)} records to database")
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
    
    def update_predictions(self) -> None:
        """Update predictions table with latest model predictions"""
        if not self.model:
            return
        
        try:
            # Get recent data for features
            query = """
            SELECT * FROM energy_consumption 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC
            """
            recent_data = pd.read_sql(query, self.engine)
            
            if recent_data.empty:
                return
            
            # Aggregate to get total demand
            current_demand = recent_data.groupby('timestamp')['consumption_kwh'].sum().iloc[0]
            
            # Prepare features for prediction (simplified)
            features = {
                'hour_of_day': datetime.now().hour,
                'temperature': recent_data['temperature'].iloc[0],
                'humidity': recent_data['humidity'].iloc[0],
                'is_weekend': recent_data['is_weekend'].iloc[0],
                'day_of_week': datetime.now().weekday(),
                'month': datetime.now().month,
                'prev_hour_consumption': current_demand,
                'prev_day_same_hour': current_demand  # Simplified
            }
            
            # Make prediction
            features_df = pd.DataFrame([features])
            predicted_demand = self.model.predict(features_df)[0]
            
            # Save prediction
            prediction_data = {
                'timestamp': datetime.now(),
                'predicted_demand': round(predicted_demand, 2),
                'actual_demand': round(current_demand, 2),
                'model_version': 'v1.0'
            }
            
            pd.DataFrame([prediction_data]).to_sql(
                'predictions', 
                self.engine, 
                if_exists='append', 
                index=False
            )
            logger.info(f"Updated predictions: {predicted_demand:.2f} kWh")
            
        except Exception as e:
            logger.error(f"Prediction update error: {e}")
    
    def clean_old_data(self, days_to_keep: int = 30) -> None:
        """Remove data older than specified days"""
        try:
            with self.engine.connect() as conn:
                # Clean energy consumption data
                result = conn.execute(
                    text(f"""
                    DELETE FROM energy_consumption 
                    WHERE timestamp < NOW() - INTERVAL '{days_to_keep} days'
                    """)
                )
                conn.commit()
                logger.info(f"Cleaned {result.rowcount} old consumption records")
                
                # Clean predictions data
                result = conn.execute(
                    text(f"""
                    DELETE FROM predictions 
                    WHERE timestamp < NOW() - INTERVAL '{days_to_keep} days'
                    """)
                )
                conn.commit()
                logger.info(f"Cleaned {result.rowcount} old prediction records")
                
        except Exception as e:
            logger.error(f"Data cleanup error: {e}")
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the pipeline"""
        try:
            with self.engine.connect() as conn:
                # Total records
                total_records = conn.execute(
                    text("SELECT COUNT(*) FROM energy_consumption")
                ).scalar()
                
                # Records in last hour
                recent_records = conn.execute(
                    text("""
                    SELECT COUNT(*) FROM energy_consumption 
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                    """)
                ).scalar()
                
                # Average consumption last hour
                avg_consumption = conn.execute(
                    text("""
                    SELECT AVG(consumption_kwh) FROM energy_consumption 
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                    """)
                ).scalar()
                
                return {
                    'total_records': total_records,
                    'recent_records': recent_records,
                    'avg_recent_consumption': round(avg_consumption, 2) if avg_consumption else 0
                }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}
    
    def run_pipeline(self) -> None:
        """Run the complete data pipeline"""
        logger.info("Starting data pipeline run...")
        
        try:
            # 1. Fetch weather data
            weather = self.fetch_weather_data()
            logger.info(f"Weather: {weather}")
            
            # 2. Generate consumption data
            consumption_df = self.generate_consumption_data(weather)
            
            # 3. Save to database
            self.save_to_database(consumption_df)
            
            # 4. Update predictions
            self.update_predictions()
            
            # 5. Clean old data (run less frequently)
            if datetime.now().hour == 0:  # Run at midnight
                self.clean_old_data()
            
            # 6. Log statistics
            stats = self.get_pipeline_stats()
            logger.info(f"Pipeline stats: {stats}")
            
            logger.info("Pipeline run completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

def create_scheduler(pipeline: EnergyDataPipeline, interval_minutes: int = 60):
    """Create and configure the scheduler"""
    scheduler = BlockingScheduler()
    
    # Schedule the main pipeline
    scheduler.add_job(
        func=pipeline.run_pipeline,
        trigger="interval",
        minutes=interval_minutes,
        id='data_pipeline',
        name='Energy Data Pipeline',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300  # 5 minutes grace time
    )
    
    # Run immediately on start
    pipeline.run_pipeline()
    
    return scheduler

def main():
    """Main entry point"""
    # Configuration
    INTERVAL_MINUTES = 60  # Run every hour
    
    # Create pipeline
    pipeline = EnergyDataPipeline()  # Remove the parameters
    
    # Create scheduler
    scheduler = create_scheduler(pipeline, INTERVAL_MINUTES)
    
    logger.info(f"Data pipeline started. Running every {INTERVAL_MINUTES} minutes...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Shutting down pipeline...")
        scheduler.shutdown()
        logger.info("Pipeline stopped")

if __name__ == "__main__":
    main()