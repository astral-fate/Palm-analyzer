import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import calendar
import os
import io

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Palm Farm Analytics Dashboard",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Your CSS styles remain the same */
    .main-header {
        font-size: 2.8rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .insight-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        border: 1px solid #b3d9ff;
        color: #0c5aa6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# --- NEW DATA LOADING FUNCTION ---
@st.cache_data
def load_all_farm_data(root_folder='Data/'):
    """
    Loads and consolidates historical data from a directory of farm folders.
    Each farm folder should contain a CSV file like '[farm_name]_historical_2015_2025.csv'.
    """
    all_dataframes = []
    
    if not os.path.isdir(root_folder):
        st.error(f"Error: The data directory '{root_folder}' was not found. Please create it and place your farm folders inside.")
        return pd.DataFrame()

    farm_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    if not farm_folders:
        st.error(f"No farm subdirectories found in the '{root_folder}' directory.")
        return pd.DataFrame()

    for farm_name in farm_folders:
        file_path = os.path.join(root_folder, farm_name, f"{farm_name}_historical_2015_2025.csv")
        
        if os.path.exists(file_path):
            try:
                temp_df = pd.read_csv(file_path)
                if 'farm_name' not in temp_df.columns:
                    temp_df['farm_name'] = farm_name
                all_dataframes.append(temp_df)
            except Exception as e:
                st.warning(f"Could not read or process file for {farm_name}: {e}")
        else:
            st.warning(f"Warning: Expected data file not found for farm '{farm_name}' at path '{file_path}'")

    if not all_dataframes:
        st.error("No valid data files could be loaded. Please check the file structure and names.")
        return pd.DataFrame()
        
    df = pd.concat(all_dataframes, ignore_index=True)
    
    # Data Cleaning and Preparation
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'farm_id' not in df.columns:
        df['farm_id'] = df['farm_name']
    
    return df


# --- Analytics and Helper Functions (No changes needed here) ---
class PalmFarmAnalytics:
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
    
    def prepare_data(self):
        self.df_indexed = self.df.set_index('timestamp')
        farm_data = []
        for farm_id in self.df['farm_id'].unique():
            farm_df = self.df[self.df['farm_id'] == farm_id].copy()
            farm_df = farm_df.sort_values('timestamp')
            farm_df['ndvi_ma_30'] = farm_df['NDVI'].rolling(window=30, min_periods=5).mean()
            farm_df['ndvi_ma_90'] = farm_df['NDVI'].rolling(window=90, min_periods=10).mean()
            farm_df['ndvi_std_30'] = farm_df['NDVI'].rolling(window=30, min_periods=5).std()
            farm_df['ndvi_trend'] = farm_df['NDVI'].diff(7)
            farm_df['ndvi_vs_avg'] = farm_df['NDVI'] - farm_df['NDVI'].mean()
            farm_data.append(farm_df)
        self.df_enhanced = pd.concat(farm_data)

# All other functions (create_overview_metrics, create_seasonal_analysis, etc.) remain the same.

# --- [Paste all your other functions here: create_overview_metrics, create_seasonal_analysis, create_farm_comparison, create_anomaly_detection] ---


def main():
    st.markdown('<h1 class="main-header">🌴 Palm Farm Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # --- MODIFIED: Load your actual data ---
    with st.spinner("Loading all farm data..."):
        df = load_all_farm_data('Data/')
        
        # Stop execution if data loading failed
        if df.empty:
            return
            
        analytics = PalmFarmAnalytics(df)
    
    # --- Sidebar Configuration ---
    st.sidebar.header("📊 Dashboard Configuration")
    
    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Farm selection (now dynamically populated from your data)
    all_farms = sorted(df['farm_name'].unique())
    selected_farms = st.sidebar.multiselect(
        "Select farms:",
        options=all_farms,
        default=all_farms,
        help="Choose which farms to include in the analysis"
    )
    
    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df['timestamp'].dt.date >= start_date) & 
            (df['timestamp'].dt.date <= end_date) &
            (df['farm_name'].isin(selected_farms)) # Filter by farm_name
        ]
    else:
        df_filtered = df[df['farm_name'].isin(selected_farms)]
    
    if df_filtered.empty:
        st.error("No data available for the selected filters.")
        return
    
    # --- Main Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🌱 Seasonal Analysis",
        "🏆 Farm Comparison", 
        "📈 Trends & Forecasting",
        "🚨 Anomaly Detection",
        "📋 Data Explorer"
    ])

    # --- [Paste the code for all your tabs (tab1, tab2, ..., tab6) here] ---
    # The code inside the tabs does not need to change, as it operates on the
    # `df_filtered` DataFrame, which is now populated with your real data.
    # Make sure to update 'M' to 'ME' for pandas frequency and fix use_container_width.
    # Example for tab1:

    with tab1:
        st.header("📊 Portfolio Overview")
        # The rest of your tab1 code...
        # Ensure to replace 'farm_id' with 'farm_name' if you are grouping by name.
        # For example, in the line chart:
        monthly_data = df_filtered.groupby([
            df_filtered['timestamp'].dt.to_period('ME'), 'farm_name'
        ])['NDVI'].mean().reset_index()
        # ... rest of the plotting code
        
if __name__ == "__main__":
    main()
