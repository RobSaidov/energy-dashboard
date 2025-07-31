# ml_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime

def prepare_features(df):
    """Prepare features for ML model"""
    
    # Aggregate to hourly total consumption
    hourly_demand = df.groupby(['timestamp', 'hour_of_day', 'temperature', 'humidity', 'is_weekend']).agg({
        'consumption_kwh': 'sum'
    }).reset_index()
    
    # Add time-based features
    hourly_demand['day_of_week'] = pd.to_datetime(hourly_demand['timestamp']).dt.dayofweek
    hourly_demand['month'] = pd.to_datetime(hourly_demand['timestamp']).dt.month
    
    # Add lag features (previous hour's consumption)
    hourly_demand['prev_hour_consumption'] = hourly_demand['consumption_kwh'].shift(1)
    hourly_demand['prev_day_same_hour'] = hourly_demand['consumption_kwh'].shift(24)
    
    # Fill NaN values
    hourly_demand = hourly_demand.bfill()
    
    return hourly_demand

def train_model(df):
    """Train Random Forest model for peak demand prediction"""
    
    # Prepare features
    df_features = prepare_features(df)
    
    # Define features and target
    feature_cols = ['hour_of_day', 'temperature', 'humidity', 'is_weekend', 
                   'day_of_week', 'month', 'prev_hour_consumption', 'prev_day_same_hour']
    
    X = df_features[feature_cols]
    y = df_features['consumption_kwh']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f} kWh")
    print(f"R² Score: {r2:.2f}")
    print(f"Accuracy: {(1 - mae/y_test.mean()) * 100:.1f}%")
    
    # Save model
    joblib.dump(model, 'energy_demand_model.pkl')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('energy_data.csv')
    
    # Train model
    model = train_model(df)