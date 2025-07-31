
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_energy_data(num_days=30, households=100):
    """Generate realistic energy consumption data"""
    
    data = []
    start_date = datetime.now() - timedelta(days=num_days)
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        for hour in range(24):
            timestamp = current_date + timedelta(hours=hour)
            
            # Base consumption patterns
            if 6 <= hour <= 9 or 17 <= hour <= 22:  # Peak hours
                base_consumption = np.random.normal(2.5, 0.5)
            elif 23 <= hour or hour <= 5:  # Night hours
                base_consumption = np.random.normal(0.8, 0.2)
            else:  # Day hours
                base_consumption = np.random.normal(1.5, 0.3)
            
            # Weather effects (simulated)
            temperature = 20 + 10 * np.sin((day * 24 + hour) / 24 * np.pi / 6) + np.random.normal(0, 2)
            humidity = 50 + 20 * np.sin((day * 24 + hour) / 24 * np.pi / 8) + np.random.normal(0, 5)
            
            # Temperature effect on consumption (AC/heating)
            if temperature > 28 or temperature < 10:
                base_consumption *= 1.3
            
            # Weekend effect
            is_weekend = timestamp.weekday() >= 5
            if is_weekend:
                base_consumption *= 1.1
            
            # Generate data for each household
            for household in range(households):
                consumption = max(0, base_consumption + np.random.normal(0, 0.2))
                
                data.append({
                    'timestamp': timestamp,
                    'household_id': f'HH_{household:03d}',
                    'consumption_kwh': round(consumption, 3),
                    'temperature': round(temperature, 1),
                    'humidity': round(humidity, 1),
                    'is_weekend': is_weekend,
                    'hour_of_day': hour
                })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('energy_data.csv', index=False)
    print(f"Generated {len(df)} records")
    
    # Also save a smaller sample for testing
    df.head(1000).to_csv('energy_data_sample.csv', index=False)
    
    return df

if __name__ == "__main__":
    df = generate_energy_data()
    print(df.head())
    print(f"\nDataset size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")