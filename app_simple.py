# app_simple.py - Start with this simpler version
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import joblib

# Page config
st.set_page_config(
    page_title="Energy Consumption Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_db_connection():
    # Update with your password
    engine = create_engine('postgresql://postgres:312413@localhost/energy_dashboard')
    return engine

# Load ML model
@st.cache_resource
def load_model():
    return joblib.load('energy_demand_model.pkl')

# Load data
@st.cache_data(ttl=300)
def load_data():
    engine = get_db_connection()
    query = "SELECT * FROM energy_consumption ORDER BY timestamp DESC LIMIT 10000"
    df = pd.read_sql(query, engine)
    return df

def main():
    st.title("⚡ Energy Consumption Analytics Dashboard")
    
    # Load data
    try:
        df = load_data()
        st.success(f"Loaded {len(df)} records from database")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure to run load_data_to_db.py first!")
        return
    
    # Show basic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        avg_consumption = df['consumption_kwh'].mean()
        st.metric("Avg Consumption", f"{avg_consumption:.2f} kWh")
    
    with col3:
        total_households = df['household_id'].nunique()
        st.metric("Households", total_households)
    
    # Simple line chart
    st.subheader("Recent Consumption")
    hourly_data = df.groupby('timestamp')['consumption_kwh'].sum().reset_index()
    fig = px.line(hourly_data.head(100), x='timestamp', y='consumption_kwh')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())

if __name__ == "__main__":
    main()