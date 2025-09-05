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

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Palm Farm Analytics Dashboard",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

@st.cache_data
def generate_sample_data():
    """Generate comprehensive sample data for demonstration"""
    np.random.seed(42)
    
    # Create date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='5D')
    
    # Farm configurations
    farms = [
        {'id': 'farm_001', 'name': 'Al-Madina North', 'base_ndvi': 0.72, 'variability': 0.15},
        {'id': 'farm_002', 'name': 'Al-Madina South', 'base_ndvi': 0.68, 'variability': 0.18},
        {'id': 'farm_003', 'name': 'Wadi Al-Furaat', 'base_ndvi': 0.75, 'variability': 0.12},
        {'id': 'farm_004', 'name': 'Oasis Valley', 'base_ndvi': 0.65, 'variability': 0.20},
        {'id': 'farm_005', 'name': 'Desert Edge', 'base_ndvi': 0.58, 'variability': 0.25}
    ]
    
    all_data = []
    
    for farm in farms:
        for date in dates:
            # Add seasonal patterns (Saudi climate)
            month = date.month
            seasonal_factor = 1.0
            
            # Hot summer stress (June-September)
            if month in [6, 7, 8, 9]:
                seasonal_factor = 0.85 - (month - 6) * 0.05
            # Good growing periods (Nov-April)
            elif month in [11, 12, 1, 2, 3, 4]:
                seasonal_factor = 1.15
            # Transition periods
            else:
                seasonal_factor = 1.0
            
            # Base NDVI with seasonal adjustment
            base_ndvi = farm['base_ndvi'] * seasonal_factor
            
            # Add random variation
            ndvi = base_ndvi + np.random.normal(0, farm['variability'] * 0.3)
            ndvi = np.clip(ndvi, 0.1, 0.95)  # Realistic NDVI range
            
            # Calculate related indices
            ndwi = -0.3 - 0.5 * ndvi + np.random.normal(0, 0.1)
            savi = ndvi * 1.5 + np.random.normal(0, 0.05)
            
            # Cloud coverage (random but realistic)
            cloud_percent = np.random.exponential(8)  # Most days low cloud, occasional high
            cloud_percent = np.clip(cloud_percent, 0, 95)
            
            # Calculate time features
            day_of_year = date.timetuple().tm_yday
            week_of_year = date.isocalendar()[1]
            quarter = (month - 1) // 3 + 1
            
            # Season mapping for Saudi climate
            if month in [12, 1, 2]:
                season = 'Winter'
            elif month in [3, 4, 5]:
                season = 'Spring'
            elif month in [6, 7, 8]:
                season = 'Summer'
            else:
                season = 'Fall'
            
            all_data.append({
                'time': int(date.timestamp() * 1000),  # Unix timestamp in milliseconds
                'farm_id': farm['id'],
                'farm_name': farm['name'],
                'NDVI': round(ndvi, 6),
                'NDWI': round(ndwi, 6),
                'SAVI': round(savi, 6),
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'day_of_year': day_of_year,
                'week_of_year': week_of_year,
                'quarter': quarter,
                'season': season,
                'cloud_percent': round(cloud_percent, 2)
            })
    
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    
    return df

class PalmFarmAnalytics:
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data with additional features"""
        # Set timestamp as index for time series operations
        self.df_indexed = self.df.set_index('timestamp')
        
        # Calculate rolling averages for each farm
        farm_data = []
        for farm_id in self.df['farm_id'].unique():
            farm_df = self.df[self.df['farm_id'] == farm_id].copy()
            farm_df = farm_df.sort_values('timestamp')
            
            # Rolling statistics
            farm_df['ndvi_ma_30'] = farm_df['NDVI'].rolling(window=30, min_periods=5).mean()
            farm_df['ndvi_ma_90'] = farm_df['NDVI'].rolling(window=90, min_periods=10).mean()
            farm_df['ndvi_std_30'] = farm_df['NDVI'].rolling(window=30, min_periods=5).std()
            
            # Trend indicators
            farm_df['ndvi_trend'] = farm_df['NDVI'].diff(7)  # 7-day change
            
            # Performance vs historical average
            farm_df['ndvi_vs_avg'] = farm_df['NDVI'] - farm_df['NDVI'].mean()
            
            farm_data.append(farm_df)
        
        self.df_enhanced = pd.concat(farm_data)

def create_overview_metrics(df):
    """Create key performance metrics"""
    total_observations = len(df)
    farms_count = df['farm_id'].nunique()
    date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
    avg_ndvi = df['NDVI'].mean()
    data_quality = (100 - df['cloud_percent'].mean())
    
    return {
        'total_observations': total_observations,
        'farms_count': farms_count,
        'date_range': date_range,
        'avg_ndvi': avg_ndvi,
        'data_quality': data_quality
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
    
    # Flatten column names
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
    
    return monthly_stats

def create_farm_comparison(df):
    """Compare farm performance"""
    farm_stats = df.groupby(['farm_id', 'farm_name']).agg({
        'NDVI': ['mean', 'std', 'min', 'max', 'count'],
        'NDWI': 'mean',
        'SAVI': 'mean',
        'cloud_percent': 'mean'
    }).round(4)
    
    # Flatten column names
    farm_stats.columns = ['_'.join(col).strip() for col in farm_stats.columns]
    
    # Calculate performance score
    farm_stats['health_score'] = (
        farm_stats['NDVI_mean'] * 0.6 +
        (1 - farm_stats['NDVI_std']) * 0.3 +
        (1 - farm_stats['cloud_percent_mean']/100) * 0.1
    ) * 100
    
    # Add ranking
    farm_stats['rank'] = farm_stats['health_score'].rank(ascending=False, method='dense').astype(int)
    
    return farm_stats.sort_values('health_score', ascending=False)

def create_anomaly_detection(df):
    """Simple anomaly detection based on statistical thresholds"""
    anomalies = []
    
    for farm_id in df['farm_id'].unique():
        farm_data = df[df['farm_id'] == farm_id].copy()
        
        # Calculate statistical thresholds
        q25 = farm_data['NDVI'].quantile(0.25)
        q75 = farm_data['NDVI'].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        # Find anomalies
        farm_anomalies = farm_data[
            (farm_data['NDVI'] < lower_bound) | 
            (farm_data['NDVI'] > upper_bound)
        ].copy()
        
        # Classify severity
        farm_anomalies['severity'] = 'Medium'
        farm_anomalies.loc[farm_anomalies['NDVI'] < q25 - 2*iqr, 'severity'] = 'High'
        farm_anomalies.loc[farm_anomalies['NDVI'] < q25 - 3*iqr, 'severity'] = 'Critical'
        
        anomalies.append(farm_anomalies)
    
    if anomalies:
        return pd.concat(anomalies)
    return pd.DataFrame()

def main():
    st.markdown('<h1 class="main-header">🌴 Palm Farm Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading farm data..."):
        df = generate_sample_data()
        analytics = PalmFarmAnalytics(df)
    
    # Sidebar configuration
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
    
    # Farm selection
    all_farms = df['farm_id'].unique()
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
            (df['farm_id'].isin(selected_farms))
        ]
    else:
        df_filtered = df[df['farm_id'].isin(selected_farms)]
    
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
        
        # Key metrics
        metrics = create_overview_metrics(df_filtered)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Observations", f"{metrics['total_observations']:,}")
        
        with col2:
            st.metric("Active Farms", f"{metrics['farms_count']}")
        
        with col3:
            st.metric("Avg NDVI", f"{metrics['avg_ndvi']:.3f}")
        
        with col4:
            st.metric("Data Quality", f"{metrics['data_quality']:.1f}%")
        
        with col5:
            years_span = df_filtered['year'].nunique()
            st.metric("Years of Data", f"{years_span}")
        
        # Portfolio performance chart
        st.subheader("📈 Portfolio Performance Over Time")
        
        # Monthly aggregation for cleaner visualization
        monthly_data = df_filtered.groupby([
            df_filtered['timestamp'].dt.to_period('M'), 'farm_id', 'farm_name'
        ])['NDVI'].mean().reset_index()
        monthly_data['timestamp'] = monthly_data['timestamp'].dt.to_timestamp()
        
        fig_portfolio = px.line(
            monthly_data,
            x='timestamp',
            y='NDVI',
            color='farm_name',
            title='Monthly NDVI Trends by Farm',
            labels={'timestamp': 'Date', 'NDVI': 'NDVI Value', 'farm_name': 'Farm'}
        )
        fig_portfolio.update_layout(height=500, hovermode='x unified')
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Performance distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df_filtered,
                x='NDVI',
                nbins=30,
                title='NDVI Distribution Across All Farms',
                color_discrete_sequence=['#2E8B57']
            )
            fig_hist.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by season
            fig_box = px.box(
                df_filtered,
                x='season',
                y='NDVI',
                title='NDVI Distribution by Season',
                color='season',
                color_discrete_map={
                    'Spring': '#90EE90',
                    'Summer': '#FFD700', 
                    'Fall': '#DEB887',
                    'Winter': '#87CEEB'
                }
            )
            fig_box.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Key insights
        st.subheader("🎯 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            best_farm = df_filtered.groupby('farm_name')['NDVI'].mean().idxmax()
            best_ndvi = df_filtered.groupby('farm_name')['NDVI'].mean().max()
            st.write(f"**🏆 Top Performer**")
            st.write(f"{best_farm}")
            st.write(f"Average NDVI: {best_ndvi:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            seasonal_avg = df_filtered.groupby('season')['NDVI'].mean()
            best_season = seasonal_avg.idxmax()
            st.write(f"**🌱 Best Season**")
            st.write(f"{best_season}")
            st.write(f"Average NDVI: {seasonal_avg[best_season]:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            recent_data = df_filtered[df_filtered['timestamp'] >= df_filtered['timestamp'].max() - timedelta(days=90)]
            recent_avg = recent_data['NDVI'].mean()
            overall_avg = df_filtered['NDVI'].mean()
            trend = "↗️ Improving" if recent_avg > overall_avg else "↘️ Declining" if recent_avg < overall_avg else "→ Stable"
            st.write(f"**📈 Recent Trend (90d)**")
            st.write(f"{trend}")
            st.write(f"Recent avg: {recent_avg:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("🌱 Comprehensive Seasonal Analysis")
        
        # Monthly statistics
        monthly_stats = create_seasonal_analysis(df_filtered)
        
        # Create month names for display
        month_names = [calendar.month_name[i] for i in range(1, 13)]
        
        # Seasonal patterns visualization
        fig_seasonal = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly NDVI Patterns', 
                'Seasonal Variability (Std Dev)',
                'Data Quality by Month',
                'Cloud Coverage Patterns'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Monthly NDVI average
        fig_seasonal.add_trace(
            go.Scatter(
                x=month_names,
                y=monthly_stats['NDVI_mean'].values,
                mode='lines+markers',
                name='Avg NDVI',
                line=dict(width=3, color='#2E8B57'),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Monthly NDVI variability
        fig_seasonal.add_trace(
            go.Bar(
                x=month_names,
                y=monthly_stats['NDVI_std'].values,
                name='NDVI Std Dev',
                marker_color='orange',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Data points by month
        fig_seasonal.add_trace(
            go.Bar(
                x=month_names,
                y=monthly_stats['NDVI_count'].values,
                name='Observations',
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Cloud coverage
        fig_seasonal.add_trace(
            go.Scatter(
                x=month_names,
                y=monthly_stats['cloud_percent_mean'].values,
                mode='lines+markers',
                name='Avg Cloud %',
                line=dict(width=2, color='gray'),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig_seasonal.update_layout(height=800, title_text="Comprehensive Seasonal Analysis")
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Seasonal insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🍂 Challenging Periods")
            lowest_months = monthly_stats['NDVI_mean'].nsmallest(3)
            
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            for month_idx, ndvi_val in lowest_months.items():
                month_name = calendar.month_name[month_idx]
                st.write(f"**{month_name}**: {ndvi_val:.3f} NDVI")
            st.write("\n**Recommendations:**")
            st.write("• Increase irrigation frequency")
            st.write("• Monitor for heat stress")
            st.write("• Consider shade protection")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("🌱 Peak Growing Periods")
            highest_months = monthly_stats['NDVI_mean'].nlargest(3)
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            for month_idx, ndvi_val in highest_months.items():
                month_name = calendar.month_name[month_idx]
                st.write(f"**{month_name}**: {ndvi_val:.3f} NDVI")
            st.write("\n**Opportunities:**")
            st.write("• Optimal for harvesting")
            st.write("• Schedule maintenance in low periods")
            st.write("• Plan expansion activities")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed monthly table
        st.subheader("📊 Detailed Monthly Statistics")
        display_stats = monthly_stats[['NDVI_mean', 'NDVI_std', 'NDVI_min', 'NDVI_max', 'NDVI_count', 'cloud_percent_mean']]
        display_stats.columns = ['Mean NDVI', 'Std Dev', 'Min NDVI', 'Max NDVI', 'Observations', 'Avg Cloud %']
        display_stats.index = month_names
        st.dataframe(display_stats.round(3), use_container_width=True)
    
    with tab3:
        st.header("🏆 Farm Performance Comparison")
        
        # Farm comparison metrics
        farm_stats = create_farm_comparison(df_filtered)
        
        # Top performers section
        st.subheader("🥇 Performance Leaderboard")
        
        col1, col2, col3 = st.columns(3)
        
        top_3 = farm_stats.head(3)
        medals = ["🥇", "🥈", "🥉"]
        
        for i, (idx, farm) in enumerate(top_3.iterrows()):
            farm_id, farm_name = idx
            with [col1, col2, col3][i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write(f"**{medals[i]} {farm_name}**")
                st.write(f"Health Score: {farm['health_score']:.1f}")
                st.write(f"Avg NDVI: {farm['NDVI_mean']:.3f}")
                st.write(f"Stability: {1/farm['NDVI_std']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance comparison chart
        farm_stats_reset = farm_stats.reset_index()
        farm_stats_reset['farm_display'] = farm_stats_reset['farm_name']
        
        fig_comparison = px.bar(
            farm_stats_reset,
            x='health_score',
            y='farm_display',
            orientation='h',
            title="Farm Health Score Comparison",
            color='health_score',
            color_continuous_scale='RdYlGn',
            text='health_score'
        )
        fig_comparison.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_comparison.update_layout(
            height=max(400, len(farm_stats) * 40),
            xaxis_title="Health Score",
            yaxis_title="Farm"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance vs Stability
            fig_scatter = px.scatter(
                farm_stats_reset,
                x='NDVI_mean',
                y='NDVI_std',
                size='NDVI_count',
                hover_name='farm_name',
                title="Performance vs Stability Analysis",
                labels={
                    'NDVI_mean': 'Average NDVI (Performance)',
                    'NDVI_std': 'NDVI Std Dev (Risk)',
                    'NDVI_count': 'Data Points'
                },
                color='health_score',
                color_continuous_scale='RdYlGn'
            )
            fig_scatter.add_annotation(
                x=farm_stats_reset['NDVI_mean'].min(),
                y=farm_stats_reset['NDVI_std'].max(),
                text="High Risk<br>Low Performance",
                showarrow=False,
                font=dict(color="red", size=10)
            )
            fig_scatter.add_annotation(
                x=farm_stats_reset['NDVI_mean'].max(),
                y=farm_stats_reset['NDVI_std'].min(),
                text="Low Risk<br>High Performance",
                showarrow=False,
                font=dict(color="green", size=10)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Monthly performance heatmap
            monthly_farm_data = df_filtered.groupby(['farm_name', 'month'])['NDVI'].mean().unstack(fill_value=0)
            
            fig_heatmap = px.imshow(
                monthly_farm_data.values,
                x=[calendar.month_abbr[i] for i in monthly_farm_data.columns],
                y=monthly_farm_data.index,
                title="Monthly Performance Heatmap",
                color_continuous_scale='RdYlGn',
                aspect="auto"
            )
            fig_heatmap.update_layout(
                xaxis_title="Month",
                yaxis_title="Farm",
                height=400
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Farm Metrics")
        display_cols = ['rank', 'health_score', 'NDVI_mean', 'NDVI_std', 'NDVI_min', 'NDVI_max', 'NDVI_count']
        display_names = ['Rank', 'Health Score', 'Avg NDVI', 'Std Dev', 'Min NDVI', 'Max NDVI', 'Data Points']
        
        comparison_table = farm_stats[display_cols].copy()
        comparison_table.columns = display_names
        comparison_table_reset = comparison_table.reset_index()
        comparison_table_reset = comparison_table_reset.drop('farm_id', axis=1)
        comparison_table_reset = comparison_table_reset.round(3)
        
        st.dataframe(comparison_table_reset, use_container_width=True, hide_index=True)
    
    with tab4:
        st.header("📈 Trends & Forecasting")
        
        # Farm selection for detailed analysis
        selected_farm_trend = st.selectbox(
            "Select farm for trend analysis:",
            options=df_filtered['farm_id'].unique(),
            format_func=lambda x: df_filtered[df_filtered['farm_id']==x]['farm_name'].iloc[0]
        )
        
        farm_trend_data = df_filtered[df_filtered['farm_id'] == selected_farm_trend].copy()
        farm_name = farm_trend_data['farm_name'].iloc[0]
        
        # Time series decomposition
        st.subheader(f"📊 Time Series Analysis: {farm_name}")
        
        # Monthly aggregation for trend analysis
        monthly_trend = farm_trend_data.groupby(farm_trend_data['timestamp'].dt.to_period('M')).agg({
            'NDVI': 'mean',
            'NDWI': 'mean',
            'SAVI': 'mean',
            'cloud_percent': 'mean'
        })
        monthly_trend.index = monthly_trend.index.to_timestamp()
        
        # Calculate trend
        if len(monthly_trend) > 12:
            # Simple linear trend
            x = np.arange(len(monthly_trend))
            trend_coef = np.polyfit(x, monthly_trend['NDVI'], 1)[0]
            trend_direction = "Improving" if trend_coef > 0.001 else "Declining" if trend_coef < -0.001 else "Stable"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trend Direction", trend_direction, f"{trend_coef*12:.4f} NDVI/year")
            with col2:
                recent_avg = monthly_trend['NDVI'].tail(6).mean()
                st.metric("Recent 6-Month Avg", f"{recent_avg:.3f}")
            with col3:
                volatility = monthly_trend['NDVI'].std()
                stability = "High" if volatility < 0.05 else "Medium" if volatility < 0.1 else "Low"
                st.metric("Stability", stability, f"σ={volatility:.3f}")
        
        # Trend visualization
        fig_trend = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'NDVI Trend Over Time',
                'Seasonal Decomposition',
                'Rolling Averages',
                'Performance Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Main trend line
        fig_trend.add_trace(
            go.Scatter(
                x=monthly_trend.index,
                y=monthly_trend['NDVI'],
                mode='lines+markers',
                name='Monthly NDVI',
                line=dict(width=2, color='#2E8B57')
            ),
            row=1, col=1
        )
        
        # Add trend line
        if len(monthly_trend) > 6:
            x_trend = np.arange(len(monthly_trend))
            y_trend = np.poly1d(np.polyfit(x_trend, monthly_trend['NDVI'], 1))(x_trend)
            fig_trend.add_trace(
                go.Scatter(
                    x=monthly_trend.index,
                    y=y_trend,
                    mode='lines',
                    name='Trend Line',
                    line=dict(width=2, color='red', dash='dash')
                ),
                row=1, col=1
            )
        
        # Seasonal pattern
        seasonal_pattern = farm_trend_data.groupby('month')['NDVI'].mean()
        month_names_short = [calendar.month_abbr[i] for i in seasonal_pattern.index]
        fig_trend.add_trace(
            go.Bar(
                x=month_names_short,
                y=seasonal_pattern.values,
                name='Seasonal Pattern',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Rolling averages
        if len(farm_trend_data) > 30:
            farm_trend_sorted = farm_trend_data.sort_values('timestamp')
            ma_30 = farm_trend_sorted['NDVI'].rolling(window=30, min_periods=5).mean()
            ma_90 = farm_trend_sorted['NDVI'].rolling(window=90, min_periods=10).mean()
            
            fig_trend.add_trace(
                go.Scatter(
                    x=farm_trend_sorted['timestamp'],
                    y=ma_30,
                    mode='lines',
                    name='30-day MA',
                    line=dict(width=1, color='blue'),
                    showlegend=False
                ),
                row=2, col=1
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=farm_trend_sorted['timestamp'],
                    y=ma_90,
                    mode='lines',
                    name='90-day MA',
                    line=dict(width=2, color='orange'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Performance distribution
        fig_trend.add_trace(
            go.Histogram(
                x=farm_trend_data['NDVI'],
                nbinsx=20,
                name='NDVI Distribution',
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig_trend.update_layout(height=800, title_text=f"Comprehensive Trend Analysis: {farm_name}")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Yearly comparison
        st.subheader("📊 Year-over-Year Comparison")
        yearly_comparison = farm_trend_data.groupby('year')['NDVI'].agg(['mean', 'std', 'count']).round(3)
        yearly_comparison.columns = ['Average NDVI', 'Standard Deviation', 'Observations']
        
        # Calculate year-over-year change
        yearly_comparison['YoY Change'] = yearly_comparison['Average NDVI'].pct_change().round(3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(yearly_comparison, use_container_width=True)
        
        with col2:
            fig_yearly = px.bar(
                x=yearly_comparison.index,
                y=yearly_comparison['Average NDVI'],
                title="Average NDVI by Year",
                color=yearly_comparison['Average NDVI'],
                color_continuous_scale='RdYlGn'
            )
            fig_yearly.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_yearly, use_container_width=True)
    
    with tab5:
        st.header("🚨 Anomaly Detection")
        
        # Detect anomalies
        anomalies_df = create_anomaly_detection(df_filtered)
        
        if not anomalies_df.empty:
            # Anomaly summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_anomalies = len(anomalies_df)
                st.metric("Total Anomalies", total_anomalies)
            
            with col2:
                critical_count = len(anomalies_df[anomalies_df['severity'] == 'Critical'])
                st.metric("Critical Issues", critical_count)
            
            with col3:
                recent_anomalies = len(anomalies_df[anomalies_df['timestamp'] >= anomalies_df['timestamp'].max() - timedelta(days=90)])
                st.metric("Recent (90d)", recent_anomalies)
            
            with col4:
                affected_farms = anomalies_df['farm_id'].nunique()
                st.metric("Affected Farms", affected_farms)
            
            # Anomaly timeline
            st.subheader("📈 Anomaly Timeline")
            
            fig_anomaly = go.Figure()
            
            # Normal data points (sample for performance)
            normal_sample = df_filtered[~df_filtered.index.isin(anomalies_df.index)].sample(
                min(1000, len(df_filtered) // 10)
            )
            
            fig_anomaly.add_trace(go.Scatter(
                x=normal_sample['timestamp'],
                y=normal_sample['NDVI'],
                mode='markers',
                name='Normal',
                marker=dict(color='lightgreen', size=4, opacity=0.6)
            ))
            
            # Anomaly points by severity
            severity_colors = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow'}
            for severity in ['Critical', 'High', 'Medium']:
                severity_data = anomalies_df[anomalies_df['severity'] == severity]
                if len(severity_data) > 0:
                    fig_anomaly.add_trace(go.Scatter(
                        x=severity_data['timestamp'],
                        y=severity_data['NDVI'],
                        mode='markers',
                        name=f'{severity} Anomaly',
                        marker=dict(
                            color=severity_colors[severity],
                            size=8,
                            symbol='x',
                            line=dict(width=1, color='black')
                        )
                    ))
            
            fig_anomaly.update_layout(
                title="Anomaly Detection Timeline",
                xaxis_title="Date",
                yaxis_title="NDVI",
                height=500,
                hovermode='closest'
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly analysis by farm and time
            col1, col2 = st.columns(2)
            
            with col1:
                # Anomalies by farm
                farm_anomaly_count = anomalies_df.groupby('farm_name').size().sort_values(ascending=False)
                fig_farm_anomaly = px.bar(
                    x=farm_anomaly_count.values,
                    y=farm_anomaly_count.index,
                    orientation='h',
                    title="Anomalies by Farm",
                    color=farm_anomaly_count.values,
                    color_continuous_scale='Reds'
                )
                fig_farm_anomaly.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_farm_anomaly, use_container_width=True)
            
            with col2:
                # Anomalies by month
                anomalies_df['month_name'] = anomalies_df['month'].apply(lambda x: calendar.month_abbr[x])
                monthly_anomalies = anomalies_df.groupby('month_name').size()
                
                fig_monthly_anomaly = px.bar(
                    x=monthly_anomalies.index,
                    y=monthly_anomalies.values,
                    title="Anomalies by Month",
                    color=monthly_anomalies.values,
                    color_continuous_scale='Oranges'
                )
                fig_monthly_anomaly.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_monthly_anomaly, use_container_width=True)
            
            # Recent critical anomalies table
            st.subheader("🔥 Recent Critical Issues")
            recent_critical = anomalies_df[
                (anomalies_df['severity'] == 'Critical') &
                (anomalies_df['timestamp'] >= anomalies_df['timestamp'].max() - timedelta(days=180))
            ].sort_values('timestamp', ascending=False)
            
            if len(recent_critical) > 0:
                display_critical = recent_critical[['timestamp', 'farm_name', 'NDVI', 'season', 'cloud_percent']].head(10)
                display_critical.columns = ['Date', 'Farm', 'NDVI', 'Season', 'Cloud %']
                st.dataframe(display_critical.round(3), use_container_width=True, hide_index=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                worst_farm = recent_critical.groupby('farm_name').size().idxmax()
                worst_season = recent_critical.groupby('season').size().idxmax()
                
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.write(f"**Priority Actions:**")
                st.write(f"• Immediate inspection needed for **{worst_farm}**")
                st.write(f"• Extra monitoring during **{worst_season}** season")
                st.write(f"• Review irrigation and fertilization schedules")
                st.write(f"• Consider soil testing for affected areas")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("🎉 No recent critical anomalies detected!")
        else:
            st.success("🎉 No anomalies detected in the selected data!")
            st.info("This indicates stable and healthy vegetation across all farms.")
    
    with tab6:
        st.header("📋 Data Explorer")
        
        # Data overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Structure:**")
            st.write(f"• Shape: {df_filtered.shape[0]:,} rows × {df_filtered.shape[1]} columns")
            st.write(f"• Date range: {df_filtered['timestamp'].min().strftime('%Y-%m-%d')} to {df_filtered['timestamp'].max().strftime('%Y-%m-%d')}")
            st.write(f"• Farms: {df_filtered['farm_id'].nunique()}")
            st.write(f"• Years covered: {df_filtered['year'].nunique()}")
        
        with col2:
            st.write("**Data Quality:**")
            completeness = (1 - df_filtered.isnull().sum() / len(df_filtered)) * 100
            st.write(f"• NDVI completeness: {completeness['NDVI']:.1f}%")
            st.write(f"• NDWI completeness: {completeness['NDWI']:.1f}%")
            st.write(f"• SAVI completeness: {completeness['SAVI']:.1f}%")
            st.write(f"• Average cloud cover: {df_filtered['cloud_percent'].mean():.1f}%")
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        numeric_columns = ['NDVI', 'NDWI', 'SAVI', 'cloud_percent']
        summary_stats = df_filtered[numeric_columns].describe().round(3)
        st.dataframe(summary_stats, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Correlation Analysis")
        correlation_matrix = df_filtered[numeric_columns].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            title="Correlation Matrix of Vegetation Indices"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Raw data viewer
        st.subheader("🔍 Raw Data Viewer")
        
        # Filters for data exploration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view_farm = st.selectbox(
                "Select farm:",
                options=['All'] + list(df_filtered['farm_id'].unique()),
                format_func=lambda x: 'All Farms' if x == 'All' else df_filtered[df_filtered['farm_id']==x]['farm_name'].iloc[0] if x != 'All' else x
            )
        
        with col2:
            view_year = st.selectbox(
                "Select year:",
                options=['All'] + sorted(df_filtered['year'].unique())
            )
        
        with col3:
            view_season = st.selectbox(
                "Select season:",
                options=['All'] + list(df_filtered['season'].unique())
            )
        
        # Apply filters
        view_data = df_filtered.copy()
        if view_farm != 'All':
            view_data = view_data[view_data['farm_id'] == view_farm]
        if view_year != 'All':
            view_data = view_data[view_data['year'] == view_year]
        if view_season != 'All':
            view_data = view_data[view_data['season'] == view_season]
        
        # Display filtered data
        st.write(f"Showing {len(view_data):,} records")
        
        # Select columns to display
        available_columns = view_data.columns.tolist()
        default_columns = ['timestamp', 'farm_name', 'NDVI', 'NDWI', 'SAVI', 'season', 'cloud_percent']
        display_columns = st.multiselect(
            "Select columns to display:",
            options=available_columns,
            default=[col for col in default_columns if col in available_columns]
        )
        
        if display_columns:
            # Sort options
            sort_column = st.selectbox("Sort by:", options=display_columns, index=0)
            sort_ascending = st.checkbox("Ascending order", value=False)
            
            # Display data
            display_data = view_data[display_columns].sort_values(sort_column, ascending=sort_ascending)
            st.dataframe(display_data, use_container_width=True, height=400)
            
            # Download option
            csv_data = display_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name=f"palm_farm_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Footer with additional info
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌴 Palm Farm Analytics**")
        st.markdown("Advanced monitoring dashboard for palm plantation health assessment")
    
    with col2:
        st.markdown("**📊 Current Session**")
        st.markdown(f"Farms analyzed: {len(selected_farms)}")
        st.markdown(f"Data points: {len(df_filtered):,}")
        st.markdown(f"Date range: {len(date_range)} days" if len(date_range) == 2 else "Full dataset")
    
    with col3:
        st.markdown("**💡 Key Metrics**")
        if not df_filtered.empty:
            st.markdown(f"Portfolio NDVI: {df_filtered['NDVI'].mean():.3f}")
            st.markdown(f"Best farm: {df_filtered.groupby('farm_name')['NDVI'].mean().idxmax()}")
            st.markdown(f"Data quality: {(100 - df_filtered['cloud_percent'].mean()):.1f}%")

if __name__ == "__main__":
    main()
