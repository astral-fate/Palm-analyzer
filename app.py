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
    /* ... (other styles remain the same) ... */
</style>
""", unsafe_allow_html=True)


# --- 0. CONFIGURATION (USER-DEFINED COLUMN NAMES) ---
# 🚨 IMPORTANT: Update these string values to match the exact lowercase column
# names from your 'consolidated_palm_farm_data.csv' file.
FARM_NAME_COL = 'farm_name'
TIMESTAMP_COL = 'timestamp'
NDVI_COL = 'ndvi'
EVI_COL = 'evi'
NDWI_COL = 'ndwi'
SAR_VV_COL = 'sar_vv'


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
    Loads and caches the main farm dataset, robustly cleaning and finding the date column.
    """
    data_path = find_file('consolidated_palm_farm_data.csv')
    if data_path is None:
        st.stop()
        
    df = pd.read_csv(data_path)
    
    # Clean all column names to remove whitespace and make them lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # --- 💡 DEBUGGING STEP 💡 ---
    # This will print the exact column names the app sees. Run the app once,
    # copy the correct names, and paste them into the CONFIGURATION section above.
    st.warning(f"**Debugging:** The following are the exact column names found in your file after cleaning: \n{df.columns.tolist()}")
    # --- END DEBUGGING STEP ---
    
    date_col_found = None
    possible_date_cols = [TIMESTAMP_COL, 'date', 'time']
    for col in possible_date_cols:
        if col in df.columns:
            date_col_found = col
            break
            
    if date_col_found:
        df[date_col_found] = pd.to_datetime(df[date_col_found])
        if date_col_found != TIMESTAMP_COL:
            df.rename(columns={date_col_found: TIMESTAMP_COL}, inplace=True)
    else:
        st.error(f"CRITICAL ERROR: No date column found. Looked for {possible_date_cols} after cleaning headers.")
        st.stop()
    
    # Verify that essential columns exist after cleaning
    required_cols = [FARM_NAME_COL, NDVI_COL, EVI_COL, NDWI_COL, SAR_VV_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"CRITICAL ERROR: The following required columns are missing from your data file after cleaning: {missing_cols}. Please check the CONFIGURATION section in the script.")
        st.stop()

    return df

# ... (The rest of the code remains exactly the same as the previous version) ...
# --- 2. CORE ANALYTICAL FUNCTIONS ---

def get_performance_report(df, scaler, kmeans_model):
    """
    Assigns farms to performance tiers using a K-Means clustering model.
    """
    kpi_df = df.groupby(FARM_NAME_COL).agg(
        mean_ndvi=(NDVI_COL, 'mean'), mean_evi=(EVI_COL, 'mean'), std_ndvi=(NDVI_COL, 'std')
    ).reset_index().dropna()

    if kpi_df.empty:
        st.warning("Not enough data to generate performance report.")
        return pd.DataFrame(columns=[FARM_NAME_COL, 'Performance Tier', 'mean_ndvi', 'mean_evi'])

    features = kpi_df[['mean_ndvi', 'mean_evi', 'std_ndvi']]
    scaled_features = scaler.transform(features)
    kpi_df['cluster'] = kmeans_model.predict(scaled_features)
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans_model.cluster_centers_), columns=['mean_ndvi', 'mean_evi', 'std_ndvi'])
    sorted_clusters = cluster_centers.sort_values(by='mean_ndvi', ascending=False).index
    tier_map = {sorted_clusters[0]: 'Tier 1 (High)', sorted_clusters[1]: 'Tier 2 (Medium)', sorted_clusters[2]: 'Tier 3 (Low)'}
    kpi_df['Performance Tier'] = kpi_df['cluster'].map(tier_map)
    return kpi_df[[FARM_NAME_COL, 'Performance Tier', 'mean_ndvi', 'mean_evi']].sort_values('Performance Tier')

def detect_and_classify_anomalies(df, farm_name):
    """
    Detects and classifies anomalies in NDVI data for a specific farm.
    """
    farm_data = df[df[FARM_NAME_COL] == farm_name].set_index(TIMESTAMP_COL).sort_index()
    df_resampled = farm_data[[NDVI_COL, NDWI_COL, SAR_VV_COL]].resample('W').mean().interpolate(method='linear')
    df_change = df_resampled.diff().dropna()

    if df_change.empty:
        st.warning("Not enough data to detect anomalies for this farm.")
        return pd.DataFrame(), go.Figure().update_layout(title=f'Not enough data for {farm_name}')

    rolling_std = df_change.rolling(window=12, min_periods=4).std()
    thresholds = {NDVI_COL: rolling_std[NDVI_COL] * 1.5, NDWI_COL: rolling_std[NDWI_COL] * 1.5, SAR_VV_COL: rolling_std[SAR_VV_COL] * 1.5}
    anomalies_found = []
    
    for date, row in df_change.iterrows():
        ndvi_change, ndwi_change, sar_vv_change = row[NDVI_COL], row[NDWI_COL], row[SAR_VV_COL]
        ndvi_thresh, ndwi_thresh, sar_thresh = thresholds[NDVI_COL].get(date, 0.07), thresholds[NDWI_COL].get(date, 0.07), thresholds[SAR_VV_COL].get(date, 1.0)
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
    fig.add_trace(go.Scatter(x=farm_data.index, y=farm_data[NDVI_COL], mode='lines', name='NDVI', line=dict(color='green')))
    colors = {'Harvest Event': 'red', 'Potential Drought Stress': 'orange', 'General Stress Event': 'purple'}
    
    for anomaly in anomalies_found:
        fig.add_shape(type='line', x0=anomaly['Date'], y0=0, x1=anomaly['Date'], y1=1, yref='paper',
                      line=dict(color=colors.get(anomaly['Classification']), width=2, dash='dash'))
        fig.add_annotation(x=anomaly['Date'], y=1.0, yref='paper', text=anomaly['Classification'], showarrow=False, yshift=10, font=dict(color=colors.get(anomaly['Classification'])))
    
    fig.update_layout(title=f'NDVI Timeline & Detected Anomalies for {farm_name}', xaxis_title='Date', yaxis_title='NDVI', height=500)
    
    display_anomalies = [{'Date': a['Date'].strftime('%Y-%m-%d'), 'Classification': a['Classification'], 'NDVI Change': a['NDVI Change']} for a in anomalies_found]
    return pd.DataFrame(display_anomalies), fig

def run_forecast(df, forecasting_models, farm_name):
    """
    Runs a 3-month NDVI forecast for a selected farm.
    """
    model = forecasting_models.get(farm_name)
    if not model:
        return None, None
    last_date = df[TIMESTAMP_COL].max()
    future_dates = pd.to_datetime(pd.date_range(start=last_date, periods=12, freq='W'))
    future_df = pd.DataFrame(index=future_dates)
    future_df['day_of_year'] = future_df.index.dayofyear
    farm_data = df[df[FARM_NAME_COL] == farm_name]
    future_df[EVI_COL] = farm_data[EVI_COL].iloc[-1]
    future_df[NDWI_COL] = farm_data[NDWI_COL].iloc[-1]
    predictions = model.predict(future_df[['day_of_year', EVI_COL, NDWI_COL]])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=farm_data[TIMESTAMP_COL], y=farm_data[NDVI_COL], mode='lines', name='Historical NDVI', line=dict(color='green')))
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
    
    ALL_FARMS = sorted(df_historical[FARM_NAME_COL].unique())
    FARM_COORDINATES = {
        'alia': [24.434117, 39.624376], 'abdula altazi': [24.499210, 39.661664],
        'albadr': [24.499454, 39.666633], 'alhabibah': [24.499002, 39.667079],
        'alia almadinah': [24.450111, 39.627500], 'almarbad': [24.442014, 39.628323],
        'alosba': [24.431591, 39.605149], 'abuonoq': [24.494620, 39.623123],
        'wahaa nakeel': [24.442692, 39.623028], 'wahaa 2': [24.442388, 39.621116]
    }
    # Clean up farm names in coordinates dict to match lowercase data
    FARM_COORDINATES = {k.lower(): v for k, v in FARM_COORDINATES.items()}

    farm_coords_df = pd.DataFrame.from_dict(FARM_COORDINATES, orient='index', columns=['lat', 'lon']).reset_index().rename(columns={'index': FARM_NAME_COL})
    farm_coords_df = farm_coords_df.merge(df_performance[[FARM_NAME_COL, 'Performance Tier']], on=FARM_NAME_COL, how='left')

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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Farms Analyzed", len(ALL_FARMS))
        with col2:
            st.metric("Average Portfolio NDVI", f"{df_historical[NDVI_COL].mean():.3f}")
        with col3:
            st.metric("Latest Data Point", df_historical[TIMESTAMP_COL].max().strftime('%Y-%m-%d'))

        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("##### Farm Locations by Performance Tier")
            color_map = {'Tier 1 (High)': 'green', 'Tier 2 (Medium)': 'orange', 'Tier 3 (Low)': 'red'}
            fig_map = px.scatter_mapbox(farm_coords_df, lat="lat", lon="lon", 
                                      hover_name=FARM_NAME_COL,
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
        
        farm_stats = df_historical.groupby(FARM_NAME_COL).agg(
            ndvi_mean=(NDVI_COL, 'mean'),
            ndvi_std=(NDVI_COL, 'std'),
            ndvi_count=(NDVI_COL, 'count')
        ).reset_index()
        farm_stats['health_score'] = (farm_stats['ndvi_mean'] * 0.7 + (1 - farm_stats['ndvi_std']) * 0.3) * 100
        farm_stats = farm_stats.sort_values('health_score', ascending=False)
        
        st.markdown("##### Farm Ranking by Health Score (NDVI Mean & Stability)")
        fig_comp = px.bar(farm_stats, x='health_score', y=FARM_NAME_COL, orientation='h',
                          color='health_score', color_continuous_scale='RdYlGn',
                          text='health_score')
        fig_comp.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_comp.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("##### Seasonal NDVI Patterns (Portfolio Average)")
        if TIMESTAMP_COL in df_historical.columns:
            df_historical['month'] = df_historical[TIMESTAMP_COL].dt.month
            monthly_avg = df_historical.groupby('month')[NDVI_COL].mean().reset_index()
            fig_seasonal = px.line(monthly_avg, x='month', y=NDVI_COL, markers=True,
                                   labels={'month': 'Month of the Year', NDVI_COL: 'Average NDVI'})
            fig_seasonal.update_xaxes(dtick=1)
            st.plotly_chart(fig_seasonal, use_container_width=True)

    # --- TAB 5: Data Explorer ---
    with tab5:
        st.subheader("Raw Data Explorer")
        st.dataframe(df_historical, use_container_width=True)

if __name__ == "__main__":
    main()
