import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import numpy as np
from datetime import datetime

# Configure Streamlit page
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
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px dashed #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

def process_consolidated_data(uploaded_file):
    """Process the consolidated CSV file containing all farm data"""
    try:
        # Read the consolidated CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['farm_id', 'time', 'NDVI']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Expected columns: farm_id, time, NDVI")
            return None, None
        
        # Handle time column
        if df['time'].dtype == 'object':
            try:
                df['time'] = pd.to_datetime(df['time'])
            except:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
        else:
            df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        df = df.set_index('time')
        
        # Process data by farm
        all_farms_data = {}
        farm_performance = []
        
        unique_farms = df['farm_id'].unique()
        
        for farm_id in unique_farms:
            farm_df = df[df['farm_id'] == farm_id].copy()
            
            # Calculate KPIs for this farm
            peak_ndvi = farm_df['NDVI'].max()
            avg_ndvi = farm_df['NDVI'].mean()
            min_ndvi = farm_df['NDVI'].min()
            std_ndvi = farm_df['NDVI'].std()
            
            # Monthly aggregation
            df_monthly = farm_df.resample('M').mean()
            
            all_farms_data[farm_id] = {
                'raw_data': farm_df,
                'monthly_data': df_monthly,
                'kpis': {
                    'peak_ndvi': peak_ndvi,
                    'avg_ndvi': avg_ndvi,
                    'min_ndvi': min_ndvi,
                    'std_ndvi': std_ndvi,
                    'data_points': len(farm_df)
                }
            }
            
            farm_performance.append({
                'farm_name': farm_id,
                'avg_ndvi': avg_ndvi,
                'peak_ndvi': peak_ndvi,
                'min_ndvi': min_ndvi,
                'volatility': std_ndvi
            })
        
        return all_farms_data, farm_performance
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

def create_seasonal_analysis(all_farms_data):
    """Create seasonal analysis from consolidated data"""
    all_monthly_data = []
    
    for farm_name, data in all_farms_data.items():
        monthly_df = data['monthly_data'].copy()
        monthly_df['farm'] = farm_name
        monthly_df['month'] = monthly_df.index.month
        all_monthly_data.append(monthly_df)
    
    if all_monthly_data:
        combined_df = pd.concat(all_monthly_data)
        seasonal_avg = combined_df.groupby('month')['NDVI'].mean()
        return seasonal_avg, combined_df
    return None, None

def create_farm_timeline(farm_data, farm_name):
    """Create individual farm timeline"""
    monthly_data = farm_data['monthly_data']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('NDVI Timeline', 'NDVI Distribution by Month'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Main timeline
    fig.add_trace(
        go.Scatter(
            x=monthly_data.index,
            y=monthly_data['NDVI'],
            mode='lines+markers',
            name='NDVI',
            line=dict(width=2, color='#2E8B57'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Monthly statistics
    monthly_stats = monthly_data.copy()
    monthly_stats['month'] = monthly_stats.index.month
    monthly_avg = monthly_stats.groupby('month')['NDVI'].mean()
    
    fig.add_trace(
        go.Bar(
            x=[calendar.month_abbr[i] for i in monthly_avg.index],
            y=monthly_avg.values,
            name='Monthly Average',
            marker_color='#90EE90',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text=f"Detailed Analysis: {farm_name}",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="NDVI", row=1, col=1)
    fig.update_yaxes(title_text="Average NDVI", row=2, col=1)
    
    return fig

def create_portfolio_heatmap(all_farms_data):
    """Create a heatmap showing farm performance over time"""
    farm_monthly_data = []
    
    for farm_name, data in all_farms_data.items():
        monthly_df = data['monthly_data'].copy()
        monthly_df['farm'] = farm_name
        monthly_df['year_month'] = monthly_df.index.strftime('%Y-%m')
        farm_monthly_data.append(monthly_df[['farm', 'year_month', 'NDVI']])
    
    if farm_monthly_data:
        combined_df = pd.concat(farm_monthly_data)
        
        # Pivot for heatmap
        heatmap_data = combined_df.pivot(index='farm', columns='year_month', values='NDVI')
        
        fig = px.imshow(
            heatmap_data,
            title="Farm Performance Heatmap Over Time",
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Farm",
            height=max(400, len(heatmap_data) * 30)
        )
        
        return fig
    return None

def main():
    st.markdown('<h1 class="main-header">🌴 Palm Farm Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("📁 Upload Consolidated Farm Data")
    st.markdown("""
    **Instructions:**
    1. Upload your consolidated CSV file containing all farm data
    2. File should have columns: `farm_id`, `time`, `NDVI`
    3. Each row represents one farm's data point at a specific time
    4. Farm identification is handled automatically via `farm_id` column
    """)
    
    uploaded_file = st.file_uploader(
        "Choose consolidated CSV file",
        type=['csv'],
        help="Upload the consolidated CSV file created by the farm data consolidator script"
    )
    
    # Show expected format
    if not uploaded_file:
        st.subheader("📋 Expected CSV Format")
        sample_data = pd.DataFrame({
            'farm_id': ['farm1', 'farm1', 'farm2', 'farm2', 'farm3', 'farm3'],
            'farm_name': ['farm1', 'farm1', 'farm2', 'farm2', 'farm3', 'farm3'],
            'time': [1609459200000, 1612137600000, 1609459200000, 1612137600000, 1609459200000, 1612137600000],
            'NDVI': [0.75, 0.72, 0.68, 0.71, 0.80, 0.78]
        })
        st.dataframe(sample_data)
        st.info("Use the farm data consolidator script to create this format from your nested folders")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded file
    if uploaded_file:
        with st.spinner("Processing consolidated farm data..."):
            all_farms_data, farm_performance = process_consolidated_data(uploaded_file)
        
        if all_farms_data is None:
            return
        
        # Store data in session state
        st.session_state.farms_data = all_farms_data
        st.session_state.farm_performance = farm_performance
        
        st.success(f"✅ Successfully processed data for {len(all_farms_data)} farms!")
        
        # Show data summary
        total_records = sum(len(data['raw_data']) for data in all_farms_data.values())
        st.info(f"📊 Loaded {total_records:,} data points across {len(all_farms_data)} farms")
    
    # Check if data exists
    if 'farms_data' not in st.session_state or not st.session_state.farms_data:
        st.info("👆 Please upload your consolidated farm data file to get started")
        return
    
    all_farms_data = st.session_state.farms_data
    farm_performance = st.session_state.farm_performance
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Overview", 
        "📈 Farm Comparison", 
        "📅 Seasonal Analysis", 
        "🔍 Individual Farm Details",
        "🌡️ Performance Heatmap"
    ])
    
    with tab1:
        st.header("Portfolio Performance Overview")
        
        if farm_performance:
            df_perf = pd.DataFrame(farm_performance)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Farms", len(df_perf))
            
            with col2:
                st.metric("Portfolio Avg NDVI", f"{df_perf['avg_ndvi'].mean():.3f}")
            
            with col3:
                best_farm = df_perf.loc[df_perf['avg_ndvi'].idxmax(), 'farm_name']
                best_ndvi = df_perf['avg_ndvi'].max()
                st.metric("Best Performing Farm", best_farm, f"{best_ndvi:.3f}")
            
            with col4:
                most_stable = df_perf.loc[df_perf['volatility'].idxmin(), 'farm_name']
                stability = df_perf['volatility'].min()
                st.metric("Most Stable Farm", most_stable, f"{stability:.3f}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df_perf, 
                    x='avg_ndvi', 
                    title="Distribution of Average NDVI Across Farms",
                    color_discrete_sequence=['#2E8B57'],
                    nbins=15
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_scatter = px.scatter(
                    df_perf, 
                    x='avg_ndvi', 
                    y='volatility',
                    size='peak_ndvi',
                    hover_name='farm_name',
                    title="Risk vs Return Analysis",
                    labels={'avg_ndvi': 'Average NDVI (Performance)', 'volatility': 'Volatility (Risk)'},
                    color='avg_ndvi',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Portfolio time series
            st.subheader("📈 Portfolio Performance Timeline")
            
            # Combine all farm data for portfolio view
            portfolio_data = []
            for farm_name, data in all_farms_data.items():
                monthly_df = data['monthly_data'].copy()
                monthly_df['farm'] = farm_name
                portfolio_data.append(monthly_df)
            
            if portfolio_data:
                combined_portfolio = pd.concat(portfolio_data)
                portfolio_avg = combined_portfolio.groupby(combined_portfolio.index)['NDVI'].mean()
                
                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(go.Scatter(
                    x=portfolio_avg.index,
                    y=portfolio_avg.values,
                    mode='lines+markers',
                    name='Portfolio Average',
                    line=dict(width=3, color='#2E8B57')
                ))
                
                fig_portfolio.update_layout(
                    title="Portfolio Average NDVI Over Time",
                    xaxis_title="Date",
                    yaxis_title="Average NDVI",
                    height=400
                )
                st.plotly_chart(fig_portfolio, use_container_width=True)
            
            # Portfolio statistics
            st.subheader("📊 Portfolio Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Performance Quartiles**")
                quartiles = df_perf['avg_ndvi'].quantile([0.25, 0.5, 0.75])
                st.write(f"Q1 (25%): {quartiles[0.25]:.3f}")
                st.write(f"Median: {quartiles[0.5]:.3f}")
                st.write(f"Q3 (75%): {quartiles[0.75]:.3f}")
            
            with col2:
                st.write("**Risk Analysis**")
                high_risk = df_perf[df_perf['volatility'] > df_perf['volatility'].quantile(0.75)]
                st.write(f"High Risk Farms: {len(high_risk)}")
                if len(high_risk) > 0:
                    st.write("Farms needing attention:")
                    for farm in high_risk['farm_name'].head(3):
                        st.write(f"• {farm}")
            
            with col3:
                st.write("**Recommendations**")
                underperformers = df_perf[df_perf['avg_ndvi'] < df_perf['avg_ndvi'].quantile(0.25)]
                st.write(f"Underperforming: {len(underperformers)} farms")
                if len(underperformers) > 0:
                    st.write("Focus improvement on:")
                    for farm in underperformers['farm_name'].head(3):
                        st.write(f"• {farm}")
    
    with tab2:
        st.header("Farm Performance Comparison")
        
        if farm_performance:
            df_perf = pd.DataFrame(farm_performance)
            
            # Sorting options
            sort_by = st.selectbox("Sort farms by:", 
                                  ["Average NDVI", "Peak NDVI", "Volatility", "Farm Name"])
            
            if sort_by == "Average NDVI":
                df_sorted = df_perf.sort_values('avg_ndvi', ascending=False)
            elif sort_by == "Peak NDVI":
                df_sorted = df_perf.sort_values('peak_ndvi', ascending=False)
            elif sort_by == "Volatility":
                df_sorted = df_perf.sort_values('volatility', ascending=True)
            else:
                df_sorted = df_perf.sort_values('farm_name')
            
            # Horizontal bar chart
            fig_comparison = px.bar(
                df_sorted,
                y='farm_name',
                x='avg_ndvi',
                orientation='h',
                title="Farm Performance Ranking",
                color='avg_ndvi',
                color_continuous_scale='RdYlGn',
                text='avg_ndvi'
            )
            fig_comparison.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_comparison.update_layout(height=max(400, len(df_perf) * 25))
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Performance tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Performers")
                top_farms = df_perf.nlargest(5, 'avg_ndvi')[['farm_name', 'avg_ndvi', 'peak_ndvi']]
                top_farms.columns = ['Farm Name', 'Avg NDVI', 'Peak NDVI']
                st.dataframe(top_farms.round(3), hide_index=True)
            
            with col2:
                st.subheader("⚠️ Needs Attention")
                bottom_farms = df_perf.nsmallest(5, 'avg_ndvi')[['farm_name', 'avg_ndvi', 'min_ndvi']]
                bottom_farms.columns = ['Farm Name', 'Avg NDVI', 'Min NDVI']
                st.dataframe(bottom_farms.round(3), hide_index=True)
            
            # Multi-farm comparison chart
            st.subheader("📊 Multi-Farm Timeline Comparison")
            
            # Allow user to select farms for comparison
            selected_farms = st.multiselect(
                "Select farms to compare:",
                options=list(all_farms_data.keys()),
                default=list(all_farms_data.keys())[:5] if len(all_farms_data) > 5 else list(all_farms_data.keys())
            )
            
            if selected_farms:
                fig_multi = go.Figure()
                
                for farm in selected_farms:
                    monthly_data = all_farms_data[farm]['monthly_data']
                    fig_multi.add_trace(go.Scatter(
                        x=monthly_data.index,
                        y=monthly_data['NDVI'],
                        mode='lines+markers',
                        name=farm,
                        line=dict(width=2)
                    ))
                
                fig_multi.update_layout(
                    title="Farm Performance Comparison Over Time",
                    xaxis_title="Date",
                    yaxis_title="NDVI",
                    height=500
                )
                st.plotly_chart(fig_multi, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("📋 Complete Farm Metrics")
            detailed_df = df_perf.copy()
            detailed_df.columns = ['Farm Name', 'Average NDVI', 'Peak NDVI', 'Minimum NDVI', 'Volatility']
            st.dataframe(detailed_df.round(3), use_container_width=True)
    
    with tab3:
        st.header("Seasonal Analysis")
        
        seasonal_avg, combined_df = create_seasonal_analysis(all_farms_data)
        
        if seasonal_avg is not None:
            # Main seasonal chart
            fig_seasonal = go.Figure()
            fig_seasonal.add_trace(go.Scatter(
                x=[calendar.month_name[i] for i in seasonal_avg.index],
                y=seasonal_avg.values,
                mode='lines+markers',
                name='Portfolio Average',
                line=dict(width=3, color='#2E8B57'),
                marker=dict(size=10)
            ))
            
            fig_seasonal.update_layout(
                title="Seasonal NDVI Patterns Across All Farms",
                xaxis_title="Month",
                yaxis_title="Average NDVI",
                height=400
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Monthly box plot
            fig_box = px.box(
                combined_df.reset_index(),
                x='month',
                y='NDVI',
                title="Monthly NDVI Distribution (All Farms)",
                labels={'month': 'Month', 'NDVI': 'NDVI Value'}
            )
            
            # Update x-axis to show month names
            fig_box.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=[calendar.month_abbr[i] for i in range(1, 13)]
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🍂 Challenging Months")
                lowest_months = seasonal_avg.nsmallest(3)
                for month_num, avg_ndvi in lowest_months.items():
                    st.write(f"**{calendar.month_name[month_num]}**: {avg_ndvi:.3f}")
                st.info("These months typically show lower vegetation health across the portfolio.")
            
            with col2:
                st.subheader("🌱 Peak Growing Months")
                highest_months = seasonal_avg.nlargest(3)
                for month_num, avg_ndvi in highest_months.items():
                    st.write(f"**{calendar.month_name[month_num]}**: {avg_ndvi:.3f}")
                st.info("These months show optimal vegetation health conditions.")
            
            # Seasonal variability by farm
            st.subheader("📊 Seasonal Variability by Farm")
            
            farm_seasonal_data = []
            for farm_name, data in all_farms_data.items():
                monthly_df = data['monthly_data'].copy()
                monthly_df['month'] = monthly_df.index.month
                farm_seasonal = monthly_df.groupby('month')['NDVI'].agg(['mean', 'std']).reset_index()
                farm_seasonal['farm'] = farm_name
                farm_seasonal_data.append(farm_seasonal)
            
            if farm_seasonal_data:
                all_farm_seasonal = pd.concat(farm_seasonal_data)
                
                fig_seasonal_farms = px.line(
                    all_farm_seasonal,
                    x='month',
                    y='mean',
                    color='farm',
                    title="Seasonal Patterns by Farm",
                    labels={'mean': 'Average NDVI', 'month': 'Month'}
                )
                
                fig_seasonal_farms.update_xaxes(
                    tickmode='array',
                    tickvals=list(range(1, 13)),
                    ticktext=[calendar.month_abbr[i] for i in range(1, 13)]
                )
                st.plotly_chart(fig_seasonal_farms, use_container_width=True)
    
    with tab4:
        st.header("Individual Farm Analysis")
        
        if all_farms_data:
            # Farm selection
            selected_farm = st.selectbox(
                "Select a farm for detailed analysis:",
                options=list(all_farms_data.keys()),
                help="Choose a farm to view detailed performance metrics and timeline"
            )
            
            if selected_farm:
                farm_data = all_farms_data[selected_farm]
                
                # Farm KPIs
                st.subheader(f"📊 Performance Metrics: {selected_farm}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Peak NDVI", 
                        f"{farm_data['kpis']['peak_ndvi']:.3f}",
                        help="Highest NDVI value recorded"
                    )
                
                with col2:
                    st.metric(
                        "Average NDVI", 
                        f"{farm_data['kpis']['avg_ndvi']:.3f}",
                        help="Mean NDVI across all time periods"
                    )
                
                with col3:
                    st.metric(
                        "Minimum NDVI", 
                        f"{farm_data['kpis']['min_ndvi']:.3f}",
                        help="Lowest NDVI value recorded"
                    )
                
                with col4:
                    st.metric(
                        "Volatility (σ)", 
                        f"{farm_data['kpis']['std_ndvi']:.3f}",
                        help="Standard deviation - lower is more stable"
                    )
                
                # Performance assessment
                avg_ndvi = farm_data['kpis']['avg_ndvi']
                if avg_ndvi > 0.7:
                    st.success(f"🌟 {selected_farm} shows excellent vegetation health!")
                elif avg_ndvi > 0.5:
                    st.info(f"✓ {selected_farm} shows good vegetation health.")
                else:
                    st.warning(f"⚠️ {selected_farm} may need attention - below average health.")
                
                # Detailed timeline
                fig_timeline = create_farm_timeline(farm_data, selected_farm)
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Recent performance and trends
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Recent Trend Analysis")
                    recent_data = farm_data['monthly_data'].tail(6)
                    if len(recent_data) >= 2:
                        trend = recent_data['NDVI'].iloc[-1] - recent_data['NDVI'].iloc[0]
                        if trend > 0.05:
                            st.success(f"📈 Improving trend (+{trend:.3f})")
                        elif trend < -0.05:
                            st.warning(f"📉 Declining trend ({trend:.3f})")
                        else:
                            st.info(f"➡️ Stable trend ({trend:+.3f})")
                    
                    # Show recent monthly averages
                    st.write("**Last 6 Months:**")
                    for date, ndvi in recent_data['NDVI'].items():
                        st.write(f"{date.strftime('%Y-%m')}: {ndvi:.3f}")
                
                with col2:
                    st.subheader("📊 Performance Comparison")
                    if farm_performance:
                        farm_rank = pd.DataFrame(farm_performance).set_index('farm_name')['avg_ndvi'].rank(ascending=False)
                        current_rank = int(farm_rank[selected_farm])
                        total_farms = len(farm_performance)
                        
                        st.metric("Portfolio Ranking", f"{current_rank} of {total_farms}")
                        
                        percentile = (total_farms - current_rank + 1) / total_farms * 100
                        if percentile >= 75:
                            st.success(f"Top {percentile:.0f}% performer! 🏆")
                        elif percentile >= 50:
                            st.info(f"Above average ({percentile:.0f}th percentile)")
                        else:
                            st.warning(f"Below average ({percentile:.0f}th percentile)")
                
                # Data download
                st.subheader("📥 Export Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    raw_data_for_export = farm_data['raw_data'].copy()
                    raw_data_for_export.reset_index(inplace=True)
                    csv_data = raw_data_for_export.to_csv(index=False)
                    st.download_button(
                        label=f"Download {selected_farm} Raw Data",
                        data=csv_data,
                        file_name=f"{selected_farm}_raw_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    monthly_data_for_export = farm_data['monthly_data'].copy()
                    monthly_data_for_export.reset_index(inplace=True)
                    monthly_csv = monthly_data_for_export.to_csv(index=False)
                    st.download_button(
                        label=f"Download {selected_farm} Monthly Summary",
                        data=monthly_csv,
                        file_name=f"{selected_farm}_monthly_summary.csv",
                        mime="text/csv"
                    )
    
    with tab5:
        st.header("Performance Heatmap")
        
        # Create heatmap
        heatmap_fig = create_portfolio_heatmap(all_farms_data)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            st.subheader("📊 Correlation Analysis")
            
            # Farm correlation analysis
            correlation_data = []
            for farm_name, data in all_farms_data.items():
                monthly_df = data['monthly_data'].copy()
                monthly_df['farm'] = farm_name
                monthly_df['year_month'] = monthly_df.index.strftime('%Y-%m')
                correlation_data.append(monthly_df[['farm', 'year_month', 'NDVI']])
            
            if correlation_data:
                combined_corr = pd.concat(correlation_data)
                corr_pivot = combined_corr.pivot(index='year_month', columns='farm', values='NDVI')
                
                # Calculate correlation matrix
                correlation_matrix = corr_pivot.corr()
                
                fig_corr = px.imshow(
                    correlation_matrix,
                    title="Farm Performance Correlation Matrix",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                
                fig_corr.update_layout(
                    xaxis_title="Farm",
                    yaxis_title="Farm",
                    height=600
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Insights from correlation analysis
                st.subheader("🔍 Correlation Insights")
                
                # Find highly correlated farms
                correlation_matrix_no_diag = correlation_matrix.copy()
                np.fill_diagonal(correlation_matrix_no_diag.values, np.nan)
                
                # Get highest correlations
                corr_stack = correlation_matrix_no_diag.stack()
                high_corr = corr_stack[corr_stack > 0.8].sort_values(ascending=False)
                
                if len(high_corr) > 0:
                    st.write("**Highly Correlated Farms (>0.8):**")
                    for (farm1, farm2), corr_val in high_corr.head(5).items():
                        st.write(f"• {farm1} ↔ {farm2}: {corr_val:.3f}")
                    st.info("These farms show very similar performance patterns - they may share similar conditions or management practices.")
                else:
                    st.info("No farms show extremely high correlation (>0.8), indicating diverse performance patterns.")
    
    # Summary section
    st.markdown("---")
    
    # Download consolidated report
    if 'farms_data' in st.session_state:
        st.subheader("📥 Export Complete Portfolio Report")
        
        # Create summary report
        portfolio_summary = []
        for farm_name, data in all_farms_data.items():
            summary = {
                'farm_id': farm_name,
                'total_records': len(data['raw_data']),
                'date_range_start': data['raw_data'].index.min(),
                'date_range_end': data['raw_data'].index.max(),
                'peak_ndvi': data['kpis']['peak_ndvi'],
                'avg_ndvi': data['kpis']['avg_ndvi'],
                'min_ndvi': data['kpis']['min_ndvi'],
                'volatility': data['kpis']['std_ndvi']
            }
            portfolio_summary.append(summary)
        
        summary_df = pd.DataFrame(portfolio_summary)
        summary_csv = summary_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📊 Download Portfolio Summary Report",
                data=summary_csv,
                file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download a summary report of all farms with key metrics"
            )
        
        with col2:
            # Create detailed farm rankings
            farm_rankings = pd.DataFrame(farm_performance)
            farm_rankings['avg_ndvi_rank'] = farm_rankings['avg_ndvi'].rank(ascending=False)
            farm_rankings['volatility_rank'] = farm_rankings['volatility'].rank(ascending=True)  # Lower volatility = better rank
            farm_rankings['peak_ndvi_rank'] = farm_rankings['peak_ndvi'].rank(ascending=False)
            
            rankings_csv = farm_rankings.to_csv(index=False)
            st.download_button(
                label="🏆 Download Farm Rankings",
                data=rankings_csv,
                file_name=f"farm_rankings_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download detailed farm rankings and performance metrics"
            )
    
    # Footer with instructions
    st.markdown("---")
    st.markdown(f"""
    **Dashboard Information:**
    - Data processed from consolidated CSV format
    - Analysis covers {len(st.session_state.get('farms_data', {}))} farms
    - Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    **How to update data:**
    1. Use the farm data consolidator script to create a new consolidated CSV
    2. Upload the new file using the file uploader above
    3. The dashboard will automatically refresh with new data
    
    **Need help?**
    - Ensure your CSV has columns: farm_id, time, NDVI
    - Use the consolidator script for proper formatting
    - Check that time values are in Unix timestamp format or readable date strings
    """)

if __name__ == "__main__":
    main()
