# app.py - Full version with all features
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import create_engine
import joblib
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
import os

# Page config
st.set_page_config(
    page_title="Energy Consumption Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Custom header with your branding
st.markdown("""
<div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>⚡ Energy Analytics Dashboard</h1>
    <p style='color: #cccccc; margin: 0; font-size: 1.1rem;'>Built by Rob Saidov | Full-Stack ML Engineering Project</p>
    <p style='color: #aaaaaa; margin: 0; font-size: 0.9rem;'>🔗 <a href='mailto:robsaidov@gmail.com' style='color: #4da6ff;'>robsaidov@gmail.com</a> | 
       <a href='https://linkedin.com/in/robsaidov' style='color: #4da6ff;'>LinkedIn</a> | 
       <a href='https://github.com/robsaidov' style='color: #4da6ff;'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_db_connection():
    """Create database connection"""
    try:
        engine = create_engine('postgresql://postgres:312413@localhost:5432/energy_dashboard')
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        raise

# Load ML model
@st.cache_resource
def load_model():
    return joblib.load('energy_demand_model.pkl')

# Load data with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(days=7):
    try:
        engine = get_db_connection()
        query = f"""
        SELECT * FROM energy_consumption 
        WHERE timestamp >= CURRENT_DATE - INTERVAL '{days} days'
        ORDER BY timestamp ASC  -- Changed to ASC for proper time series
        """
        df = pd.read_sql(query, engine)
        
        if df.empty:
            st.error("No data found in database")
            return pd.DataFrame()
        
        # Ensure all required columns exist
        required_columns = ['timestamp', 'consumption_kwh', 'temperature', 'humidity', 'household_id']
        if not all(col in df.columns for col in required_columns):
            st.error("Missing required columns in data")
            return pd.DataFrame()
        
        # Convert and validate data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['consumption_kwh'] = pd.to_numeric(df['consumption_kwh'], errors='coerce')
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
        
        # Add derived columns
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Drop rows with invalid data
        df = df.dropna(subset=['consumption_kwh', 'temperature'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Anomaly detection function
def detect_anomalies(df, threshold=3):
    """Detect anomalies using z-score method"""
    df_copy = df.copy()
    df_copy['z_score'] = np.abs((df_copy['consumption_kwh'] - df_copy['consumption_kwh'].mean()) / df_copy['consumption_kwh'].std())
    anomalies = df_copy[df_copy['z_score'] > threshold]
    return anomalies

# Prepare features for prediction
def prepare_prediction_features(df, hours_ahead=24):
    """Prepare features for next 24 hours prediction"""
    last_data = df.iloc[0]  # Most recent data
    future_times = pd.date_range(start=last_data['timestamp'], periods=hours_ahead+1, freq='H')[1:]
    
    # Get average consumption for each hour
    hourly_avg = df.groupby('hour_of_day')['consumption_kwh'].mean()
    
    future_data = []
    for t in future_times:
        hour = t.hour
        # Simple temperature prediction (sine wave pattern)
        temp = 20 + 10 * np.sin((t.hour - 6) * np.pi / 12)
        
        future_data.append({
            'hour_of_day': hour,
            'temperature': temp,
            'humidity': 60,  # Average humidity
            'is_weekend': t.weekday() >= 5,
            'day_of_week': t.weekday(),
            'month': t.month,
            'prev_hour_consumption': hourly_avg.get(hour, df['consumption_kwh'].mean()),
            'prev_day_same_hour': hourly_avg.get(hour, df['consumption_kwh'].mean()),
            'timestamp': t
        })
    
    return pd.DataFrame(future_data)

def main():
    st.title("⚡ Real-Time Energy Consumption Analytics Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Date range selector
        days_to_show = st.slider("Days of data to display", 1, 30, 7)
        
        # Simplified refresh button
        if st.button("🔄 Refresh Data"):
            # Only clear data cache, not resource cache
            load_data.clear()
            st.success("Data refreshed!")
        
        st.markdown("---")
        st.markdown("**Data Updates**")
        st.info("Dashboard refreshes every 5 minutes")
        st.markdown("Last update: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Load data
    try:
        df = load_data(days_to_show)
        model = load_model()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_consumption = df['consumption_kwh'].sum()
        st.metric(
            "Total Consumption", 
            f"{total_consumption:,.0f} kWh",
            f"{total_consumption/1000:.1f} MWh"
        )
    
    with col2:
        avg_daily = df.groupby(df['timestamp'].dt.date)['consumption_kwh'].sum().mean()
        st.metric(
            "Avg Daily Usage", 
            f"{avg_daily:,.0f} kWh",
            f"↑ {(avg_daily/df['household_id'].nunique()):.1f} per household"
        )
    
    with col3:
        peak_hour = df.groupby('hour_of_day')['consumption_kwh'].sum().idxmax()
        st.metric(
            "Peak Hour", 
            f"{peak_hour}:00 - {(peak_hour+1)%24}:00",
            "High demand period"
        )
    
    with col4:
        current_temp = df.iloc[0]['temperature'] if not df.empty else 20
        st.metric(
            "Current Temp", 
            f"{current_temp:.1f}°C",
            f"{current_temp * 9/5 + 32:.1f}°F"
        )
    
    with col5:
        valid_temps = df[df['temperature'].between(18, 24)]['consumption_kwh']
        if len(valid_temps) > 0:
            efficiency = (valid_temps.mean() / df['consumption_kwh'].mean() * 100)
            efficiency_text = f"{efficiency:.0f}%"
        else:
            efficiency_text = "N/A"
            
        st.metric(
            "Efficiency Score", 
            efficiency_text,
            "Optimal temp range"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Predictions", "🌡️ Weather Analysis", "🚨 Anomalies"])
    
    with tab1:
        st.subheader("Energy Consumption Trends")
        
        # Aggregate by hour for smooth visualization
        hourly_consumption = df.groupby(pd.Grouper(key='timestamp', freq='H'))['consumption_kwh'].sum().reset_index()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=hourly_consumption['timestamp'],
            y=hourly_consumption['consumption_kwh'],
            mode='lines',
            name='Total Consumption',
            line=dict(color='#00ff00', width=2)
        ))
        
        # Add moving average
        hourly_consumption['MA24'] = hourly_consumption['consumption_kwh'].rolling(window=24).mean()
        fig_trend.add_trace(go.Scatter(
            x=hourly_consumption['timestamp'],
            y=hourly_consumption['MA24'],
            mode='lines',
            name='24h Moving Average',
            line=dict(color='#ff8c00', width=2, dash='dash')
        ))
        
        fig_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title='Consumption (kWh)'),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Two columns for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            hourly_pattern = df.groupby('hour_of_day')['consumption_kwh'].agg(['mean', 'std']).reset_index()
            
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=hourly_pattern['hour_of_day'],
                y=hourly_pattern['mean'],
                error_y=dict(type='data', array=hourly_pattern['std'], visible=True),
                marker_color='lightblue',
                name='Avg Consumption'
            ))
            
            fig_hourly.update_layout(
                title="Average Consumption by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Avg Consumption (kWh)",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Weekend vs Weekday
            weekend_comparison = df.groupby('is_weekend')['consumption_kwh'].mean().reset_index()
            weekend_comparison['day_type'] = weekend_comparison['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
            
            fig_weekend = px.pie(
                weekend_comparison, 
                values='consumption_kwh', 
                names='day_type',
                title="Weekend vs Weekday Consumption",
                color_discrete_map={'Weekend': '#ff6b6b', 'Weekday': '#4ecdc4'}
            )
            fig_weekend.update_traces(textposition='inside', textinfo='percent+label')
            fig_weekend.update_layout(height=350)
            st.plotly_chart(fig_weekend, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Peak Demand Prediction - Next 24 Hours")
        
        # Prepare prediction data
        future_df = prepare_prediction_features(df)
        
        # Make predictions
        feature_cols = ['hour_of_day', 'temperature', 'humidity', 'is_weekend', 
                       'day_of_week', 'month', 'prev_hour_consumption', 'prev_day_same_hour']
        predictions = model.predict(future_df[feature_cols])
        future_df['predicted_demand'] = predictions
        
        # Prediction chart
        fig_pred = go.Figure()
        
        # Historical data (last 24 hours)
        recent_history = hourly_consumption.tail(24)
        fig_pred.add_trace(go.Scatter(
            x=recent_history['timestamp'],
            y=recent_history['consumption_kwh'],
            mode='lines',
            name='Historical',
            line=dict(color='#00ff00', width=2)
        ))
        
        # Predictions
        fig_pred.add_trace(go.Scatter(
            x=future_df['timestamp'],
            y=future_df['predicted_demand'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#ff00ff', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Add confidence interval
        std_dev = predictions.std()
        fig_pred.add_trace(go.Scatter(
            x=future_df['timestamp'].tolist() + future_df['timestamp'].tolist()[::-1],
            y=(predictions + 1.96*std_dev).tolist() + (predictions - 1.96*std_dev).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='95% Confidence'
        ))
        
        # Mark peak predicted hours
        peak_hours = future_df.nlargest(3, 'predicted_demand')
        fig_pred.add_trace(go.Scatter(
            x=peak_hours['timestamp'],
            y=peak_hours['predicted_demand'],
            mode='markers',
            name='Peak Hours',
            marker=dict(color='red', size=15, symbol='star')
        ))
        
        fig_pred.update_layout(
            height=500,
            xaxis_title="Time",
            yaxis_title="Demand (kWh)",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Prediction summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Peak Demand", f"{predictions.max():.0f} kWh", 
                     f"at {future_df.loc[predictions.argmax(), 'timestamp'].strftime('%H:%M')}")
        with col2:
            st.metric("Predicted Min Demand", f"{predictions.min():.0f} kWh",
                     f"at {future_df.loc[predictions.argmin(), 'timestamp'].strftime('%H:%M')}")
        with col3:
            st.metric("24h Total Prediction", f"{predictions.sum():.0f} kWh",
                     f"Avg: {predictions.mean():.0f} kWh/h")
    
    with tab3:
        st.subheader("🌡️ Weather Impact Analysis")
        
        # Temperature vs Consumption scatter
        fig_temp = px.scatter(
            df.sample(min(5000, len(df))),  # Sample for performance
            x='temperature',
            y='consumption_kwh',
            color='is_weekend',
            title="Temperature vs Energy Consumption",
            labels={'consumption_kwh': 'Consumption (kWh)', 'temperature': 'Temperature (°C)'},
            color_discrete_map={True: '#ff6b6b', False: '#4ecdc4'}
        )
        
        # Replace the polyfit code with a more robust solution
        # Find this section in the Weather Analysis tab
        def robust_trendline(x, y):
            mask = ~np.isnan(x) & ~np.isnan(y)
            if len(x[mask]) > 2:
                try:
                    z = np.polyfit(x[mask], y[mask], 1)  # Use linear fit instead of quadratic
                    p = np.poly1d(z)
                    return p
                except:
                    return None
            return None

        # Use the robust function
        trend = robust_trendline(df['temperature'], df['consumption_kwh'])
        if trend is not None:
            temp_range = np.linspace(df['temperature'].min(), df['temperature'].max(), 100)
            fig_temp.add_trace(go.Scatter(
                x=temp_range,
                y=trend(temp_range),
                mode='lines',
                name='Trend',
                line=dict(color='yellow', width=3)
            ))
        
        fig_temp.update_layout(height=400)
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Temperature zones analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Define temperature zones
            df['temp_zone'] = pd.cut(df['temperature'], 
                                    bins=[-np.inf, 10, 18, 24, 30, np.inf],
                                    labels=['Very Cold', 'Cold', 'Optimal', 'Warm', 'Hot'])
            
            zone_consumption = df.groupby('temp_zone')['consumption_kwh'].mean().reset_index()
            
            fig_zones = px.bar(
                zone_consumption,
                x='temp_zone',
                y='consumption_kwh',
                title="Average Consumption by Temperature Zone",
                color='consumption_kwh',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_zones, use_container_width=True)
        
        with col2:
            # Humidity impact
            fig_humidity = px.scatter(
                df.sample(min(1000, len(df))),
                x='humidity',
                y='consumption_kwh',
                color='temperature',
                title="Humidity vs Consumption (colored by temperature)",
                color_continuous_scale='thermal'
            )
            st.plotly_chart(fig_humidity, use_container_width=True)
    
    with tab4:
        st.subheader("🚨 Anomaly Detection")
        
        # Detect anomalies
        anomalies = detect_anomalies(df)
        
        if not anomalies.empty:
            st.warning(f"⚠️ Found {len(anomalies)} anomalous readings in the past {days_to_show} days!")
            
            # Anomaly visualization
            fig_anomaly = go.Figure()
            
            # Normal consumption
            fig_anomaly.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['consumption_kwh'],
                mode='markers',
                name='Normal',
                marker=dict(color='lightblue', size=4, opacity=0.5)
            ))
            
            # Anomalies
            fig_anomaly.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['consumption_kwh'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x')
            ))
            
            # Add threshold lines
            mean_consumption = df['consumption_kwh'].mean()
            std_consumption = df['consumption_kwh'].std()
            
            fig_anomaly.add_hline(y=mean_consumption + 3*std_consumption, 
                                line_dash="dash", line_color="orange",
                                annotation_text="Upper Threshold")
            fig_anomaly.add_hline(y=mean_consumption - 3*std_consumption, 
                                line_dash="dash", line_color="orange",
                                annotation_text="Lower Threshold")
            
            fig_anomaly.update_layout(
                height=400,
                xaxis_title="Time",
                yaxis_title="Consumption (kWh)",
                hovermode='closest'
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly details
            st.subheader("Recent Anomalies")
            anomaly_display = anomalies[['timestamp', 'household_id', 'consumption_kwh', 'temperature', 'z_score']].head(10)
            anomaly_display['z_score'] = anomaly_display['z_score'].round(2)
            st.dataframe(anomaly_display, use_container_width=True)
            
            # Anomaly patterns
            col1, col2 = st.columns(2)
            with col1:
                # By hour
                anomaly_hours = anomalies['hour_of_day'].value_counts().sort_index()
                fig_anomaly_hour = px.bar(
                    x=anomaly_hours.index,
                    y=anomaly_hours.values,
                    title="Anomalies by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Count'}
                )
                st.plotly_chart(fig_anomaly_hour, use_container_width=True)
            
            with col2:
                # By household
                top_anomaly_households = anomalies['household_id'].value_counts().head(10)
                fig_anomaly_household = px.bar(
                    x=top_anomaly_households.values,
                    y=top_anomaly_households.index,
                    orientation='h',
                    title="Top 10 Households with Anomalies",
                    labels={'x': 'Anomaly Count', 'y': 'Household ID'}
                )
                st.plotly_chart(fig_anomaly_household, use_container_width=True)
        else:
            st.success("✅ No anomalies detected in recent data!")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data Source**: PostgreSQL Database")
    with col2:
        st.markdown("**Model**: Random Forest (77.6% accuracy)")
    with col3:
        st.markdown("**Update Frequency**: Every 5 minutes")

# Project information function
def show_project_info():
    st.markdown("""
    # 👨‍💻 Project Portfolio: Energy Analytics Dashboard
    
    **Developer**: Rob Saidov  
    **Email**: robsaidov@gmail.com  
    **LinkedIn**: [linkedin.com/in/robsaidov](https://linkedin.com/in/robsaidov)  
    **GitHub**: [github.com/robsaidov](https://github.com/robsaidov)

    ## 🎯 Project Overview
    This is a **production-ready** ML-powered energy analytics platform demonstrating:
    - Real-time data processing and visualization
    - Machine learning for demand forecasting (77.6% accuracy)
    - AWS cloud deployment with PostgreSQL backend
    - Interactive Streamlit dashboard with 4 analytical modules
    
    ## 🛠️ Tech Stack
    **Backend**: Python, PostgreSQL, SQLAlchemy, APScheduler  
    **ML/Analytics**: scikit-learn, pandas, numpy  
    **Frontend**: Streamlit, Plotly, HTML/CSS  
    **Deployment**: AWS EC2, systemd, Linux administration
    
    ---
    """)

# About button in sidebar
if st.sidebar.button("👤 About This Project"):
    show_project_info()
else:
    if __name__ == "__main__":
        main()