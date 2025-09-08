import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import calendarimport streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import calendar
import io
import os
import joblib # Added for model loading

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Palm Farm Intelligence Platform",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (no changes from original)
st.markdown("""
<style>
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
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .insight-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        border: 1px solid #b3d9ff;
        color: #0c5aa6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { height: 60px; padding-left: 24px; padding-right: 24px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px 12px 0 0; color: #495057; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%); color: white; }
    .dataframe { border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)


# --- NEW: Functions for loading models and robust file finding ---
def find_file(filename, search_paths=['./', './models/']):
    for path in search_paths:
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            return filepath
    return None

@st.cache_resource
def load_models():
    """Loads all required .joblib models."""
    scaler_path = find_file('scaler.joblib')
    kmeans_path = find_file('kmeans_model.joblib')
    forecasting_path = find_file('forecasting_models.joblib')
    
    if not all([scaler_path, kmeans_path, forecasting_path]):
        st.error("Could not find all required model files (.joblib). Please ensure 'scaler.joblib', 'kmeans_model.joblib', and 'forecasting_models.joblib' are in the project directory.")
        return None, None, None
        
    try:
        scaler = joblib.load(scaler_path)
        kmeans_model = joblib.load(kmeans_path)
        forecasting_models = joblib.load(forecasting_path)
        return scaler, kmeans_model, forecasting_models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_real_data(folder_path):
    """Loads and consolidates historical data from a directory of farm folders."""
    all_farm_data = []
    if not os.path.isdir(folder_path):
        st.error(f"Error: The directory '{folder_path}' was not found.")
        return pd.DataFrame()

    farm_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not farm_folders:
        st.error(f"No farm folders found in the '{folder_path}' directory.")
        return pd.DataFrame()

    for farm_name in farm_folders:
        farm_dir = os.path.join(folder_path, farm_name)
        try:
            target_file = next(f for f in os.listdir(farm_dir) if f.endswith('_historical_2015_2025.csv'))
            file_path = os.path.join(farm_dir, target_file)
            df_farm = pd.read_csv(file_path)
            df_farm['farm_name'] = farm_name
            all_farm_data.append(df_farm)
        except StopIteration:
            st.warning(f"Warning: No historical CSV file found in folder: {farm_name}")
        except Exception as e:
            st.error(f"Error loading data for farm '{farm_name}': {e}")
    
    if not all_farm_data:
        st.error("No valid data could be loaded.")
        return pd.DataFrame()
        
    consolidated_df = pd.concat(all_farm_data, ignore_index=True)
    consolidated_df['timestamp'] = pd.to_datetime(consolidated_df['timestamp'])
    
    # MODIFIED: Ensure all required columns for EDA and models exist
    required_cols = ['NDVI', 'NDWI', 'EVI', 'SAVI', 'SAR_VV', 'cloud_percent', 'season', 'month', 'year', 'farm_name']
    for col in required_cols:
        if col not in consolidated_df.columns:
            consolidated_df[col] = 0
            st.warning(f"Column '{col}' was missing for models/EDA and has been filled with 0. Results may be inaccurate.")

    return consolidated_df

# --- EDA Functions (largely unchanged) ---
def create_overview_metrics(df):
    """Create key performance metrics"""
    total_observations = len(df)
    farms_count = df['farm_name'].nunique()
    date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
    avg_ndvi = df['NDVI'].mean()
    data_quality = (100 - df['cloud_percent'].mean())
    return {'total_observations': total_observations, 'farms_count': farms_count, 'date_range': date_range, 'avg_ndvi': avg_ndvi, 'data_quality': data_quality}

def create_seasonal_analysis(df):
    """Analyze seasonal patterns"""
    monthly_stats = df.groupby('month').agg({'NDVI': ['mean', 'std', 'min', 'max', 'count'], 'NDWI': 'mean', 'SAVI': 'mean', 'cloud_percent': 'mean'}).round(4)
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
    return monthly_stats

def create_farm_comparison(df):
    """Compare farm performance (EDA version)"""
    farm_stats = df.groupby('farm_name').agg({'NDVI': ['mean', 'std', 'min', 'max', 'count'], 'NDWI': 'mean', 'SAVI': 'mean', 'cloud_percent': 'mean'}).round(4)
    farm_stats.columns = ['_'.join(col).strip() for col in farm_stats.columns]
    farm_stats['health_score'] = (farm_stats['NDVI_mean'] * 0.6 + (1 - farm_stats['NDVI_std']) * 0.3 + (1 - farm_stats['cloud_percent_mean']/100) * 0.1) * 100
    farm_stats['rank'] = farm_stats['health_score'].rank(ascending=False, method='dense').astype(int)
    return farm_stats.sort_values('health_score', ascending=False)

# --- NEW: Functions adapted from Gradio Project ---
def get_performance_tiers(df, scaler, kmeans_model):
    """Calculates AI-powered performance tiers using K-Means clustering."""
    if 'EVI' not in df.columns:
        st.warning("EVI column not found for Performance Tiering. Using zeros.")
        df['EVI'] = 0
        
    kpi_df = df.groupby('farm_name').agg(mean_ndvi=('NDVI', 'mean'), mean_evi=('EVI', 'mean'), std_ndvi=('NDVI', 'std')).reset_index().dropna()
    
    if kpi_df.empty:
        return pd.DataFrame(columns=['farm_name', 'Performance Tier', 'mean_ndvi'])

    features = kpi_df[['mean_ndvi', 'mean_evi', 'std_ndvi']]
    scaled_features = scaler.transform(features)
    kpi_df['cluster'] = kmeans_model.predict(scaled_features)
    
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans_model.cluster_centers_), columns=['mean_ndvi', 'mean_evi', 'std_ndvi'])
    sorted_clusters = cluster_centers.sort_values(by='mean_ndvi', ascending=False).index
    tier_map = {sorted_clusters[0]: 'Tier 1 (High)', sorted_clusters[1]: 'Tier 2 (Medium)', sorted_clusters[2]: 'Tier 3 (Low)'}
    kpi_df['Performance Tier'] = kpi_df['cluster'].map(tier_map)
    
    return kpi_df[['farm_name', 'Performance Tier', 'mean_ndvi']].sort_values('Performance Tier')

def run_advanced_anomaly_detection(df, farm_name):
    """Detects and classifies anomalies using dynamic thresholds."""
    farm_data = df[df['farm_name'] == farm_name].set_index('timestamp').sort_index()
    if farm_data.empty:
        return pd.DataFrame(), go.Figure().update_layout(title="No data for this farm.")
        
    df_resampled = farm_data[['NDVI', 'NDWI', 'SAR_VV']].resample('W').mean().interpolate(method='linear')
    df_change = df_resampled.diff().dropna()
    rolling_std = df_change.rolling(window=12, min_periods=4).std()
    
    anomalies_found = []
    for date, row in df_change.iterrows():
        ndvi_change, ndwi_change, sar_vv_change = row['NDVI'], row['NDWI'], row['SAR_VV']
        ndvi_thresh = rolling_std['NDVI'].get(date, 0.07) * 1.5
        ndwi_thresh = rolling_std['NDWI'].get(date, 0.07) * 1.5
        sar_thresh = rolling_std['SAR_VV'].get(date, 1.0) * 1.5
        
        classification = "Normal"
        if ndvi_change < -ndvi_thresh and sar_vv_change < -sar_thresh:
            classification = 'Harvest Event'
        elif ndvi_change < -ndvi_thresh and ndwi_change < -ndwi_thresh:
            classification = 'Potential Drought Stress'
        elif ndvi_change < -ndvi_thresh:
            classification = 'General Stress Event'
        
        if classification != "Normal":
            anomalies_found.append({'Date': date, 'Classification': classification, 'NDVI Change': f"{ndvi_change:.3f}"})
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=farm_data.index, y=farm_data['NDVI'], mode='lines', name='NDVI', line=dict(color='green')))
    colors = {'Harvest Event': 'red', 'Potential Drought Stress': 'orange', 'General Stress Event': 'purple'}
    
    for anomaly in anomalies_found:
        fig.add_vline(x=anomaly['Date'], line_width=2, line_dash="dash", line_color=colors.get(anomaly['Classification']),
                      annotation_text=anomaly['Classification'], annotation_position="top left")

    fig.update_layout(title=f'NDVI Timeline & Detected Anomalies for {farm_name}', xaxis_title='Date', yaxis_title='NDVI', height=500)
    
    display_anomalies = pd.DataFrame(anomalies_found)
    if not display_anomalies.empty:
        display_anomalies['Date'] = display_anomalies['Date'].dt.strftime('%Y-%m-%d')
        
    return display_anomalies, fig

def run_forecast(df, farm_name, forecasting_models):
    """Generates a 12-week NDVI forecast for a selected farm."""
    model = forecasting_models.get(farm_name)
    if not model:
        st.error(f"No forecast model found for '{farm_name}'.")
        return None, None

    farm_data = df[df['farm_name'] == farm_name]
    last_date = farm_data['timestamp'].max()
    
    future_dates = pd.to_datetime(pd.date_range(start=last_date, periods=13, freq='W'))[1:]
    future_df = pd.DataFrame(index=future_dates)
    future_df['day_of_year'] = future_df.index.dayofyear
    future_df['EVI'] = farm_data['EVI'].iloc[-1]
    future_df['NDWI'] = farm_data['NDWI'].iloc[-1]
    
    predictions = model.predict(future_df[['day_of_year', 'EVI', 'NDWI']])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=farm_data['timestamp'], y=farm_data['NDVI'], mode='lines', name='Historical NDVI'))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Forecasted NDVI', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'12-Week NDVI Forecast for {farm_name}', height=500)
    
    forecast_table = pd.DataFrame({'Forecast Date': future_dates.strftime('%Y-%m-%d'), 'Predicted NDVI': np.round(predictions, 3)})
    return fig, forecast_table

def main():
    st.markdown('<h1 class="main-header">🌴 Palm Farm Intelligence Platform</h1>', unsafe_allow_html=True)
    
    # Load data and models
    scaler, kmeans_model, forecasting_models = load_models()
    
    with st.spinner("Loading farm data..."):
        df = load_real_data("Data")
        if df.empty:
            st.error("Data loading failed. Please check the 'Data' folder and its contents.")
            return
    
    # Sidebar configuration
    st.sidebar.header("📊 Dashboard Configuration")
    min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
    date_range = st.sidebar.date_input("Select date range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    all_farms = sorted(df['farm_name'].unique())
    selected_farms = st.sidebar.multiselect("Select farms:", options=all_farms, default=all_farms)
    
    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date) & (df['farm_name'].isin(selected_farms))]
    else:
        df_filtered = df[df['farm_name'].isin(selected_farms)]
    
    if df_filtered.empty:
        st.error("No data available for the selected filters.")
        return
    
    # Create main tabs
    tabs = st.tabs(["📊 Overview", "🌱 Seasonal Analysis", "🏆 Farm Comparison", "📈 Trends & Forecasting", "🚨 Anomaly Detection", "📋 Data Explorer"])
    
    # --- TAB 1: OVERVIEW (Unchanged) ---
    with tabs[0]:
        st.header("📊 Portfolio Overview")
        metrics = create_overview_metrics(df_filtered)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Observations", f"{metrics['total_observations']:,}")
        c2.metric("Active Farms", f"{metrics['farms_count']}")
        c3.metric("Avg NDVI", f"{metrics['avg_ndvi']:.3f}")
        c4.metric("Data Quality", f"{metrics['data_quality']:.1f}%")
        c5.metric("Years of Data", f"{df_filtered['year'].nunique()}")
        
        st.subheader("📈 Portfolio Performance Over Time")
        monthly_data = df_filtered.groupby([df_filtered['timestamp'].dt.to_period('M'), 'farm_name'])['NDVI'].mean().reset_index()
        monthly_data['timestamp'] = monthly_data['timestamp'].dt.to_timestamp()
        fig_portfolio = px.line(monthly_data, x='timestamp', y='NDVI', color='farm_name', title='Monthly NDVI Trends by Farm')
        st.plotly_chart(fig_portfolio, use_container_width=True)
        # ... (rest of the overview tab is the same)
        st.subheader("🎯 Key Insights")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            best_farm = df_filtered.groupby('farm_name')['NDVI'].mean().idxmax()
            best_ndvi = df_filtered.groupby('farm_name')['NDVI'].mean().max()
            st.write(f"**🏆 Top Performer**<br>{best_farm}<br>Average NDVI: {best_ndvi:.3f}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            seasonal_avg = df_filtered.groupby('season')['NDVI'].mean()
            best_season = seasonal_avg.idxmax()
            st.write(f"**🌱 Best Season**<br>{best_season}<br>Average NDVI: {seasonal_avg.max():.3f}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            recent_data = df_filtered[df_filtered['timestamp'] >= df_filtered['timestamp'].max() - timedelta(days=90)]
            recent_avg = recent_data['NDVI'].mean()
            overall_avg = df_filtered['NDVI'].mean()
            trend = "↗️ Improving" if recent_avg > overall_avg else "↘️ Declining"
            st.write(f"**📈 Recent Trend (90d)**<br>{trend}<br>Recent avg: {recent_avg:.3f}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: SEASONAL ANALYSIS (Unchanged) ---
    with tabs[1]:
        st.header("🌱 Comprehensive Seasonal Analysis")
        # ... (code for this tab remains identical)
        monthly_stats = create_seasonal_analysis(df_filtered)
        st.subheader("📊 Detailed Monthly Statistics")
        st.dataframe(monthly_stats.rename(columns={'NDVI_mean': 'Mean NDVI', 'NDVI_std': 'Std Dev'}).style.background_gradient(cmap='Greens', subset=['NDVI_mean']), use_container_width=True)

    # --- TAB 3: FARM COMPARISON (MODIFIED) ---
    with tabs[2]:
        st.header("🏆 Farm Performance Comparison")
        farm_stats = create_farm_comparison(df_filtered)
        st.subheader("🥇 Performance Leaderboard (Health Score)")
        # ... (existing leaderboard code is fine)
        fig_comparison = px.bar(farm_stats.reset_index(), x='health_score', y='farm_name', orientation='h', title="Farm Health Score Comparison", color='health_score', color_continuous_scale='RdYlGn', text='health_score')
        st.plotly_chart(fig_comparison, use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 AI-Powered Performance Tiers")
        if kmeans_model:
            tier_df = get_performance_tiers(df_filtered, scaler, kmeans_model)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("Farms are categorized into tiers using K-Means clustering on key performance indicators (Mean NDVI, Mean EVI, NDVI Stability).")
                tier_counts = tier_df['Performance Tier'].value_counts()
                fig_pie = px.pie(tier_counts, values=tier_counts.values, names=tier_counts.index, title="Farm Distribution by Tier", color=tier_counts.index, color_discrete_map={'Tier 1 (High)':'green', 'Tier 2 (Medium)':'orange', 'Tier 3 (Low)':'red'})
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                st.dataframe(tier_df, use_container_width=True, hide_index=True)
        else:
            st.warning("K-Means model not loaded. AI Performance Tiers are unavailable.")

    # --- TAB 4: TRENDS & FORECASTING (MODIFIED) ---
    with tabs[3]:
        st.header("📈 Trends & Forecasting")
        selected_farm_trend = st.selectbox("Select farm for trend analysis:", options=all_farms)
        # ... (existing trend analysis code is fine)

        st.markdown("---")
        st.subheader(f"🔮 12-Week NDVI Forecast for {selected_farm_trend}")
        if forecasting_models:
            if st.button(f"Generate Forecast for {selected_farm_trend}"):
                with st.spinner("Running forecast model..."):
                    fig_forecast, forecast_table = run_forecast(df, selected_farm_trend, forecasting_models)
                    if fig_forecast:
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        else:
            st.warning("Forecasting models not loaded. Forecasts are unavailable.")

    # --- TAB 5: ANOMALY DETECTION (MODIFIED) ---
    with tabs[4]:
        st.header("🚨 Advanced Anomaly Detection")
        st.info("This tool uses dynamic thresholds based on rolling statistics to identify significant changes in vegetation health.")
        
        farm_to_scan = st.selectbox("Select a farm to scan for anomalies:", options=all_farms)
        
        if farm_to_scan:
            with st.spinner(f"Scanning {farm_to_scan} for anomalies..."):
                anomalies_df, fig_anomaly_timeline = run_advanced_anomaly_detection(df_filtered, farm_to_scan)

                st.plotly_chart(fig_anomaly_timeline, use_container_width=True)

                if not anomalies_df.empty:
                    st.subheader("Detected Anomalies")
                    st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
                else:
                    st.success(f"🎉 No significant anomalies detected for {farm_to_scan} in the selected period.")

    # --- TAB 6: DATA EXPLORER (Unchanged) ---
    with tabs[5]:
        st.header("📋 Data Explorer")
        st.subheader("📊 Dataset Overview")
        # ... (code for this tab remains identical)
        st.dataframe(df_filtered.head(100), use_container_width=True)

    # ... (Footer is unchanged)
    st.markdown("---")
    st.markdown("🌴 **Palm Farm Intelligence Platform** | Advanced monitoring for plantation health")

if __name__ == "__main__":
    main()
import io
import os
import joblib # NEW: For loading the SRA model

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Palm Farm Intelligence Hub",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
st.markdown("""
<style>
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
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .insight-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        border: 1px solid #b3d9ff;
        color: #0c5aa6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding-left: 24px;
        padding-right: 24px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px 12px 0 0;
        color: #495057;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
    }
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """NEW: Loads the pre-trained SRA model."""
    model_path = 'sra_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        st.error(f"SRA Model not found at '{model_path}'. Please ensure the model file is uploaded.")
        return None

@st.cache_data
def load_enriched_data(file_path):
    """MODIFIED: Loads the single, enriched consolidated CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Error: The data file '{file_path}' was not found.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure all required columns exist, fill with defaults if not
        required_cols = ['NDVI', 'NDWI', 'SAVI', 'cloud_percent', 'season', 'month', 'year', 'farm_name', 'is_anomaly', 'anomaly_type']
        for col in required_cols:
            if col not in df.columns:
                if col == 'is_anomaly':
                    df[col] = False
                elif col == 'anomaly_type':
                    df[col] = 'Normal'
                else:
                    df[col] = 0
                st.warning(f"Column '{col}' was missing and has been filled with default values.")
        
        return df
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {e}")
        return pd.DataFrame()

def calculate_sra(df, model):
    """NEW: Calculates Stress Risk Assessment score and category using the loaded model."""
    if model is None or df.empty:
        df['sra_score'] = 0.0
        df['sra_category'] = 'Unknown'
        return df

    # Features must match the model's training features
    features = ['NDVI', 'NDWI', 'SAVI', 'ndvi_ma_30', 'ndvi_std_30']
    
    # Check for missing features and handle them gracefully
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.warning(f"SRA calculation skipped: Missing required features {missing_features}.")
        df['sra_score'] = 0.0
        df['sra_category'] = 'Unknown'
        return df

    # Drop rows with NaNs in feature columns for prediction
    df_predict = df.dropna(subset=features)
    
    if not df_predict.empty:
        # Predict probability of being in the "stressed" class (class 1)
        sra_scores = model.predict_proba(df_predict[features])[:, 1]
        
        # Map scores back to the original dataframe
        df.loc[df_predict.index, 'sra_score'] = sra_scores

        # Define risk categories based on score
        conditions = [
            df['sra_score'] <= 0.3,
            (df['sra_score'] > 0.3) & (df['sra_score'] <= 0.6),
            df['sra_score'] > 0.6
        ]
        choices = ['Low', 'Medium', 'High']
        df['sra_category'] = np.select(conditions, choices, default='Unknown')
    else:
        df['sra_score'] = 0.0
        df['sra_category'] = 'Unknown'
        
    return df

def create_farm_comparison(df):
    """UPDATED: Compares farm performance using SRA and other metrics."""
    farm_stats = df.groupby('farm_name').agg({
        'NDVI': ['mean', 'std', 'min', 'max'],
        'sra_score': 'mean', # NEW: Aggregate SRA score
        'timestamp': 'count'
    }).round(4)
    
    farm_stats.columns = ['_'.join(col).strip() for col in farm_stats.columns]
    
    # NEW: Advanced performance score incorporating SRA
    farm_stats['performance_score'] = (
        farm_stats['NDVI_mean'] * 0.5 +
        (1 - farm_stats['sra_score_mean']) * 0.4 +
        (1 - farm_stats['NDVI_std']) * 0.1
    ) * 100
    
    farm_stats['rank'] = farm_stats['performance_score'].rank(ascending=False, method='dense').astype(int)
    
    return farm_stats.sort_values('performance_score', ascending=False)

# Other helper functions (create_overview_metrics, create_seasonal_analysis, etc.) remain largely the same,
# but will now operate on the enriched, filtered dataframe.
def create_overview_metrics(df):
    """Create key performance metrics"""
    total_observations = len(df)
    farms_count = df['farm_name'].nunique()
    date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
    avg_ndvi = df['NDVI'].mean()
    data_quality = (100 - df['cloud_percent'].mean())
    avg_sra = df['sra_score'].mean() if 'sra_score' in df.columns else 0.0
    
    return {
        'total_observations': total_observations,
        'farms_count': farms_count,
        'date_range': date_range,
        'avg_ndvi': avg_ndvi,
        'data_quality': data_quality,
        'avg_sra': avg_sra
    }

def create_seasonal_analysis(df):
    """Analyze seasonal patterns"""
    # Monthly averages
    monthly_stats = df.groupby('month').agg({
        'NDVI': ['mean', 'std', 'min', 'max', 'count'],
        'NDWI': 'mean',
        'SAVI': 'mean',
        'cloud_percent': 'mean'
    }).round(4)
    
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
    return monthly_stats

def main():
    st.markdown('<h1 class="main-header">🌴 Palm Farm Intelligence Hub</h1>', 
                unsafe_allow_html=True)
    
    # Load model and data
    sra_model = load_models()
    with st.spinner("Loading enriched farm data..."):
        df = load_enriched_data("consolidated_farm_data.csv")
        if df.empty:
            st.error("Data loading failed. Please check the required CSV file.")
            return
    
    # Sidebar configuration
    st.sidebar.header("📊 Dashboard Configuration")
    min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Select date range:", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    
    all_farms = sorted(df['farm_name'].unique())
    selected_farms = st.sidebar.multiselect(
        "Select farms:", options=all_farms, default=all_farms
    )
    
    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df['timestamp'].dt.date >= start_date) & 
            (df['timestamp'].dt.date <= end_date) &
            (df['farm_name'].isin(selected_farms))
        ].copy()
    else:
        df_filtered = df[df['farm_name'].isin(selected_farms)].copy()
    
    # NEW: Calculate SRA on the filtered data
    df_filtered = calculate_sra(df_filtered, sra_model)

    if df_filtered.empty:
        st.error("No data available for the selected filters.")
        return
    
    # Create main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🌱 Seasonal Analysis",
        "🏆 Farm Comparison", 
        "📈 Trends & Forecasting",
        "🚨 Anomaly Detection",
        "📋 Data Explorer"
    ])
    
    with tab1:
        st.header("📊 Portfolio Overview")
        metrics = create_overview_metrics(df_filtered)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Active Farms", f"{metrics['farms_count']}")
        with col2:
            st.metric("Avg NDVI", f"{metrics['avg_ndvi']:.3f}")
        with col3: # NEW: SRA Metric
            st.metric("Avg Stress Risk", f"{metrics['avg_sra']:.2%}", help="Model-driven score indicating the average risk of stress events.")
        with col4:
            st.metric("Data Quality", f"{metrics['data_quality']:.1f}%")
        with col5:
            st.metric("Total Observations", f"{metrics['total_observations']:,}")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("📈 Portfolio Performance Over Time")
            monthly_data = df_filtered.groupby(df_filtered['timestamp'].dt.to_period('M'))['NDVI'].mean().reset_index()
            monthly_data['timestamp'] = monthly_data['timestamp'].dt.to_timestamp()
            fig_portfolio = px.line(monthly_data, x='timestamp', y='NDVI', title='Average Monthly NDVI Across All Selected Farms')
            st.plotly_chart(fig_portfolio, use_container_width=True)

        with col2: # NEW: SRA Distribution
            st.subheader("⚠️ Stress Risk Distribution")
            sra_counts = df_filtered['sra_category'].value_counts()
            fig_sra = px.pie(
                values=sra_counts.values, 
                names=sra_counts.index, 
                title='Portfolio Stress Risk Levels',
                color=sra_counts.index,
                color_discrete_map={'Low':'green', 'Medium':'orange', 'High':'red', 'Unknown': 'grey'}
            )
            fig_sra.update_traces(textinfo='percent+label', pull=[0.05, 0.05, 0.1])
            st.plotly_chart(fig_sra, use_container_width=True)

    with tab2:
        # This tab remains functionally the same but now operates on richer data
        st.header("🌱 Comprehensive Seasonal Analysis")
        monthly_stats = create_seasonal_analysis(df_filtered)
        month_names = [calendar.month_name[i] for i in range(1, 13)]
        
        # Seasonal patterns visualization
        fig_seasonal = make_subplots(rows=1, cols=2, subplot_titles=('Monthly NDVI Patterns', 'Cloud Coverage Patterns'))
        fig_seasonal.add_trace(go.Scatter(x=month_names, y=monthly_stats['NDVI_mean'].values, mode='lines+markers', name='Avg NDVI', line=dict(width=3, color='#2E8B57')), row=1, col=1)
        fig_seasonal.add_trace(go.Scatter(x=month_names, y=monthly_stats['cloud_percent_mean'].values, mode='lines', name='Avg Cloud %', line=dict(width=2, color='gray')), row=1, col=2)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        st.subheader("📊 Detailed Monthly Statistics")
        st.dataframe(monthly_stats, use_container_width=True)

    with tab3:
        st.header("🏆 Farm Performance Comparison")
        farm_stats = create_farm_comparison(df_filtered)
        
        st.subheader("🥇 Performance Leaderboard")
        fig_comparison = px.bar(
            farm_stats.reset_index(), x='performance_score', y='farm_name', orientation='h',
            title="Farm Performance Score Comparison", color='performance_score', color_continuous_scale='RdYlGn',
            text='performance_score'
        )
        fig_comparison.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_comparison.update_layout(height=max(400, len(farm_stats) * 40))
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.subheader("📋 Detailed Farm Metrics")
        display_cols = ['rank', 'performance_score', 'NDVI_mean', 'sra_score_mean', 'NDVI_std']
        st.dataframe(farm_stats[display_cols], use_container_width=True)
        
    with tab4:
        # This tab remains functionally the same
        st.header("📈 Trends & Forecasting")
        selected_farm_trend = st.selectbox("Select farm for trend analysis:", options=df_filtered['farm_name'].unique())
        farm_trend_data = df_filtered[df_filtered['farm_name'] == selected_farm_trend]
        
        st.subheader(f"📊 Time Series Analysis: {selected_farm_trend}")
        fig_trend = px.line(farm_trend_data, x='timestamp', y='NDVI', title='NDVI Trend Over Time')
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab5: #################### ANOMALY TAB REVAMPED ####################
        st.header("🚨 Anomaly Detection & Analysis")
        
        anomalies_df = df_filtered[df_filtered['is_anomaly'] == True]
        
        if not anomalies_df.empty:
            # NEW: Filter by farm for detailed anomaly analysis
            st.sidebar.subheader("🚨 Anomaly Filters")
            anomaly_farm_filter = st.sidebar.selectbox(
                "Filter Anomalies by Farm:",
                options=['All Farms'] + all_farms,
                index=0
            )

            if anomaly_farm_filter != 'All Farms':
                anomalies_df = anomalies_df[anomalies_df['farm_name'] == anomaly_farm_filter]
            
            if anomalies_df.empty:
                st.success(f"🎉 No anomalies detected for {anomaly_farm_filter} in the selected period!")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Anomalies", len(anomalies_df))
                col2.metric("Affected Farms", anomalies_df['farm_name'].nunique())
                recent_anomalies = len(anomalies_df[anomalies_df['timestamp'] >= anomalies_df['timestamp'].max() - timedelta(days=90)])
                col3.metric("Recent (90d)", recent_anomalies)
                
                # NEW: Chart for anomaly type breakdown
                st.subheader("📊 Breakdown of Anomaly Types")
                anomaly_counts = anomalies_df['anomaly_type'].value_counts()
                fig_anomaly_pie = px.pie(
                    values=anomaly_counts.values, 
                    names=anomaly_counts.index, 
                    title='Distribution of Detected Anomalies',
                    hole=0.3
                )
                fig_anomaly_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_anomaly_pie, use_container_width=True)

                st.subheader("📈 Anomaly Timeline")
                # Plot all points for the selected farm, highlighting anomalies
                plot_data = df_filtered[df_filtered['farm_name'] == anomaly_farm_filter] if anomaly_farm_filter != 'All Farms' else df_filtered
                
                fig_anomaly_timeline = px.scatter(
                    plot_data, x='timestamp', y='NDVI',
                    color=np.where(plot_data['is_anomaly'], plot_data['anomaly_type'], 'Normal'),
                    symbol='is_anomaly',
                    title=f"NDVI Timeline with Anomalies for {anomaly_farm_filter}",
                    color_discrete_map={
                        'Normal': 'lightgreen',
                        'Drought Stress': 'orange',
                        'Harvest Event': 'red',
                        'Pest Infestation': 'purple',
                        'Nutrient Deficiency': 'brown'
                    },
                    hover_data=['farm_name', 'anomaly_type']
                )
                fig_anomaly_timeline.update_traces(marker=dict(size=5), selector=dict(name='Normal'))
                fig_anomaly_timeline.update_traces(marker=dict(size=10, symbol='x'), selector=dict(name='True'))
                st.plotly_chart(fig_anomaly_timeline, use_container_width=True)

                st.subheader("📋 Details of Recent Anomalies")
                recent_anomalies_table = anomalies_df[anomalies_df['timestamp'] >= anomalies_df['timestamp'].max() - timedelta(days=180)]
                st.dataframe(recent_anomalies_table[['timestamp', 'farm_name', 'NDVI', 'anomaly_type']].sort_values('timestamp', ascending=False), use_container_width=True)

        else:
            st.success("🎉 No anomalies detected in the selected data!")
    
    with tab6: # MODIFIED: Data Explorer with new columns
        st.header("📋 Data Explorer")
        st.write("Explore the raw, enriched dataset with advanced filters.")
        
        # Display filtered data
        available_columns = df_filtered.columns.tolist()
        # NEW: Added SRA and anomaly columns to default view
        default_columns = ['timestamp', 'farm_name', 'NDVI', 'sra_score', 'sra_category', 'is_anomaly', 'anomaly_type', 'cloud_percent']
        display_columns = st.multiselect(
            "Select columns to display:",
            options=available_columns,
            default=[col for col in default_columns if col in available_columns]
        )
        
        if display_columns:
            st.dataframe(df_filtered[display_columns], use_container_width=True, height=500)
            
            # Download option
            csv_data = df_filtered[display_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name=f"palm_farm_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
