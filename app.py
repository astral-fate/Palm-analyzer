import streamlit as st
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
        df = load_enriched_data("consolidated_farm_data_enriched.csv")
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
