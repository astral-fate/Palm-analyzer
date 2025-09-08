import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Palm Farm Intelligence Platform",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #F0F8F0;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. DATA AND MODEL LOADING (with Caching) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    """Robustly finds a file by searching in directories relative to the app's location."""
    search_paths = [
        os.path.join(SCRIPT_DIR),
        os.path.join(SCRIPT_DIR, 'Data'),
        os.path.join(SCRIPT_DIR, 'Model')
    ]
    for path in search_paths:
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            return filepath
    st.error(f"CRITICAL ERROR: Could not find '{filename}'. Looked in directories relative to the app script.")
    return None

@st.cache_resource
def load_models():
    """Loads and caches the machine learning models."""
    scaler_path = find_file('scaler.joblib')
    kmeans_path = find_file('kmeans_model.joblib')
    forecasting_path = find_file('forecasting_models.joblib')
    if not all([scaler_path, kmeans_path, forecasting_path]):
        st.stop()
    scaler = joblib.load(scaler_path)
    kmeans_model = joblib.load(kmeans_path)
    forecasting_models = joblib.load(forecasting_path)
    return scaler, kmeans_model, forecasting_models

@st.cache_data
def load_data():
    """
    Loads and caches the main farm dataset, robustly handling the date column.
    """
    data_path = find_file('consolidated_palm_farm_data.csv')
    if data_path is None:
        st.stop()
        
    df = pd.read_csv(data_path)
    
    # ✨ FIX: Automatically find and convert the date column
    date_col_found = None
    possible_date_cols = ['timestamp', 'Date', 'date']
    for col in possible_date_cols:
        if col in df.columns:
            date_col_found = col
            break
            
    if date_col_found:
        df[date_col_found] = pd.to_datetime(df[date_col_found])
        # Rename to 'timestamp' to ensure consistency across the app
        if date_col_found != 'timestamp':
            df.rename(columns={date_col_found: 'timestamp'}, inplace=True)
    else:
        st.error(f"CRITICAL ERROR: No date column found in the CSV. Looked for one of {possible_date_cols}.")
        st.stop()

    return df

# --- 2. CORE ANALYTICAL FUNCTIONS (from Gradio project) ---

def get_performance_report(df, scaler, kmeans_model):
    """Assigns farms to performance tiers using a K-Means clustering model."""
    kpi_df = df.groupby('farm_name').agg(
        mean_ndvi=('NDVI', 'mean'), mean_evi=('EVI', 'mean'), std_ndvi=('NDVI', 'std')
    ).reset_index().dropna()
    features = kpi_df[['mean_ndvi', 'mean_evi', 'std_ndvi']]
    scaled_features = scaler.transform(features)
    kpi_df['cluster'] = kmeans_model.predict(scaled_features)
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans_model.cluster_centers_), columns=['mean_ndvi', 'mean_evi', 'std_ndvi'])
    sorted_clusters = cluster_centers.sort_values(by='mean_ndvi', ascending=False).index
    tier_map = {sorted_clusters[0]: 'Tier 1 (High)', sorted_clusters[1]: 'Tier 2 (Medium)', sorted_clusters[2]: 'Tier 3 (Low)'}
    kpi_df['Performance Tier'] = kpi_df['cluster'].map(tier_map)
    return kpi_df[['farm_name', 'Performance Tier', 'mean_ndvi', 'mean_evi']].sort_values('Performance Tier')

def detect_and_classify_anomalies(df, farm_name):
    """Detects and classifies anomalies in NDVI data for a specific farm."""
    farm_data = df[df['farm_name'] == farm_name].set_index('timestamp').sort_index()
    df_resampled = farm_data[['NDVI', 'NDWI', 'SAR_VV']].resample('W').mean().interpolate(method='linear')
    df_change = df_resampled.diff().dropna()
    rolling_std = df_change.rolling(window=12, min_periods=4).std()
    thresholds = {'NDVI': rolling_std['NDVI'] * 1.5, 'NDWI': rolling_std['NDWI'] * 1.5, 'SAR_VV': rolling_std['SAR_VV'] * 1.5}
    anomalies_found = []
    for date, row in df_change.iterrows():
        ndvi_change, ndwi_change, sar_vv_change = row['NDVI'], row['NDWI'], row['SAR_VV']
        ndvi_thresh, ndwi_thresh, sar_thresh = thresholds['NDVI'].get(date, 0.07), thresholds['NDWI'].get(date, 0.07), thresholds['SAR_VV'].get(date, 1.0)
        classification = "Normal"
        if ndvi_change < -ndvi_thresh and sar_vv_change < -sar_thresh:
            classification = 'Harvest Event'
        elif ndvi_change < -ndvi_thresh and ndwi_change < -ndwi_thresh:
            classification = 'Potential Drought Stress'
        elif ndvi_change < -ndvi_thresh:
            classification = 'General Stress Event'
        if classification != "Normal":
            anomalies_found.append({'Date': date, 'Classification': classification, 'NDVI Change': f"{ndvi_change:.3f}"})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=farm_data.index, y=farm_data['NDVI'], mode='lines', name='NDVI', line=dict(color='green')))
    colors = {'Harvest Event': 'red', 'Potential Drought Stress': 'orange', 'General Stress Event': 'purple'}
    
    for anomaly in anomalies_found:
        fig.add_shape(type='line', x0=anomaly['Date'], y0=0, x1=anomaly['Date'], y1=1, yref='paper',
                      line=dict(color=colors.get(anomaly['Classification']), width=2, dash='dash'))
        fig.add_annotation(x=anomaly['Date'], y=1.0, yref='paper', text=anomaly['Classification'], showarrow=False, yshift=10, font=dict(color=colors.get(anomaly['Classification'])))
    
    fig.update_layout(title=f'NDVI Timeline & Detected Anomalies for {farm_name}', xaxis_title='Date', yaxis_title='NDVI', height=500)
    
    display_anomalies = [{'Date': a['Date'].strftime('%Y-%m-%d'), 'Classification': a['Classification'], 'NDVI Change': a['NDVI Change']} for a in anomalies_found]
    return pd.DataFrame(display_anomalies), fig

def run_forecast(df, forecasting_models, farm_name):
    """Runs a 3-month NDVI forecast for a selected farm."""
    model = forecasting_models.get(farm_name)
    if not model:
        return None, None
    last_date = df['timestamp'].max()
    future_dates = pd.to_datetime(pd.date_range(start=last_date, periods=12, freq='W'))
    future_df = pd.DataFrame(index=future_dates)
    future_df['day_of_year'] = future_df.index.dayofyear
    farm_data = df[df['farm_name'] == farm_name]
    future_df['EVI'] = farm_data['EVI'].iloc[-1]
    future_df['NDWI'] = farm_data['NDWI'].iloc[-1]
    predictions = model.predict(future_df[['day_of_year', 'EVI', 'NDWI']])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=farm_data['timestamp'], y=farm_data['NDVI'], mode='lines', name='Historical NDVI', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Forecasted NDVI', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'3-Month NDVI Forecast for {farm_name}', height=500)
    
    forecast_df = pd.DataFrame({'Forecast Date': future_dates.strftime('%Y-%m-%d'), 'Predicted NDVI': np.round(predictions, 3)})
    return forecast_df, fig

# --- 3. MAIN APPLICATION ---
def main():
    st.markdown('<h1 class="main-header">🌴 Palm Farm Intelligence Platform</h1>', unsafe_allow_html=True)
    
    # --- Load all data and models ---
    scaler, kmeans_model, forecasting_models = load_models()
    df_historical = load_data()
    df_performance = get_performance_report(df_historical, scaler, kmeans_model)
    
    ALL_FARMS = sorted(df_historical['farm_name'].unique())
    FARM_COORDINATES = {
        'alia': [24.434117, 39.624376], 'Abdula altazi': [2.4499210, 39.661664],
        'albadr': [24.499454, 39.666633], 'alhabibah': [24.499002, 39.667079],
        'alia almadinah': [24.450111, 39.627500], 'almarbad': [24.442014, 39.628323],
        'alosba': [24.431591, 39.605149], 'abuonoq': [24.494620, 39.623123],
        'wahaa nakeel': [24.442692, 39.623028], 'wahaa 2': [24.442388, 39.621116]
    }
    farm_coords_df = pd.DataFrame.from_dict(FARM_COORDINATES, orient='index', columns=['lat', 'lon']).reset_index().rename(columns={'index':'farm_name'})
    farm_coords_df = farm_coords_df.merge(df_performance[['farm_name', 'Performance Tier']], on='farm_name', how='left')

    # --- Sidebar Filters ---
    st.sidebar.header("📋 Dashboard Filters")
    selected_farm = st.sidebar.selectbox("Select a Farm for Detailed Analysis:", ALL_FARMS)
    
    # --- Main Tabs for Navigation ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌟 Performance Overview", 
        "🚨 Anomaly Detection",
        "📈 NDVI Forecasting",
        "🌱 EDA & Farm Comparison",
        "📋 Data Explorer"
    ])
    
    # --- TAB 1: Performance Overview ---
    with tab1:
        st.subheader("Portfolio Performance & Tiers")
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Farms Analyzed", len(ALL_FARMS))
        with col2:
            st.metric("Average Portfolio NDVI", f"{df_historical['NDVI'].mean():.3f}")
        with col3:
            st.metric("Latest Data Point", df_historical['timestamp'].max().strftime('%Y-%m-%d'))

        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("##### Farm Locations by Performance Tier")
            color_map = {'Tier 1 (High)': 'green', 'Tier 2 (Medium)': 'orange', 'Tier 3 (Low)': 'red'}
            fig_map = px.scatter_mapbox(farm_coords_df, lat="lat", lon="lon", 
                                      hover_name="farm_name",
                                      hover_data=["Performance Tier"],
                                      color="Performance Tier",
                                      color_discrete_map=color_map,
                                      zoom=10, height=500)
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            
        with col2:
            st.markdown("##### Farm Tier Distribution")
            tier_counts = df_performance['Performance Tier'].value_counts().reset_index()
            tier_counts.columns = ['Performance Tier', 'Count']
            fig_tier = px.bar(tier_counts, x='Performance Tier', y='Count',
                              color='Performance Tier', text_auto=True,
                              color_discrete_map=color_map)
            fig_tier.update_layout(showlegend=False)
            st.plotly_chart(fig_tier, use_container_width=True)

            st.markdown("##### Performance Tier Table")
            st.dataframe(df_performance, use_container_width=True)

    # --- TAB 2: Anomaly Detection ---
    with tab2:
        st.subheader(f"Intelligent Anomaly Detection for: **{selected_farm}**")
        st.info("This model detects significant negative changes in vegetation health, classifying them into potential causes like drought or harvesting events.")
        
        anomaly_df, anomaly_fig = detect_and_classify_anomalies(df_historical, selected_farm)
        
        if anomaly_df.empty:
            st.success("No significant anomalies detected for this farm in the historical data.")
        else:
            st.plotly_chart(anomaly_fig, use_container_width=True)
            st.markdown("##### Detected Anomaly Events")
            st.dataframe(anomaly_df, use_container_width=True)

    # --- TAB 3: NDVI Forecasting ---
    with tab3:
        st.subheader(f"3-Month Vegetation Health Forecast for: **{selected_farm}**")
        st.info("This forecast predicts the weekly NDVI value for the next 12 weeks based on historical patterns and related vegetation indices.")

        forecast_df, forecast_fig = run_forecast(df_historical, forecasting_models, selected_farm)
        
        if forecast_fig:
            st.plotly_chart(forecast_fig, use_container_width=True)
            st.markdown("##### Forecast Data")
            st.dataframe(forecast_df, use_container_width=True)
        else:
            st.error(f"No forecasting model available for {selected_farm}.")

    # --- TAB 4: EDA & Farm Comparison ---
    with tab4:
        st.subheader("Exploratory Data Analysis")
        
        # Farm Comparison (Health Score from old app)
        farm_stats = df_historical.groupby('farm_name').agg(
            NDVI_mean=('NDVI', 'mean'),
            NDVI_std=('NDVI', 'std'),
            NDVI_count=('NDVI', 'count')
        ).reset_index()
        farm_stats['health_score'] = (farm_stats['NDVI_mean'] * 0.7 + (1 - farm_stats['NDVI_std']) * 0.3) * 100
        farm_stats = farm_stats.sort_values('health_score', ascending=False)
        
        st.markdown("##### Farm Ranking by Health Score (NDVI Mean & Stability)")
        fig_comp = px.bar(farm_stats, x='health_score', y='farm_name', orientation='h',
                          color='health_score', color_continuous_scale='RdYlGn',
                          text='health_score')
        fig_comp.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_comp.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Seasonal Analysis
        st.markdown("##### Seasonal NDVI Patterns (Portfolio Average)")
        df_historical['month'] = df_historical['timestamp'].dt.month
        monthly_avg = df_historical.groupby('month')['NDVI'].mean().reset_index()
        fig_seasonal = px.line(monthly_avg, x='month', y='NDVI', markers=True,
                               labels={'month': 'Month of the Year', 'NDVI': 'Average NDVI'})
        fig_seasonal.update_xaxes(dtick=1)
        st.plotly_chart(fig_seasonal, use_container_width=True)


    # --- TAB 5: Data Explorer ---
    with tab5:
        st.subheader("Raw Data Explorer")
        st.dataframe(df_historical, use_container_width=True)

if __name__ == "__main__":
    main()

