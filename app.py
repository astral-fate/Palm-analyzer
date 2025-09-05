import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import calendar
import numpy as np
from datetime import datetime
import io
import zipfile

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

def process_uploaded_files(uploaded_files):
    """Process uploaded CSV files"""
    all_farms_data = {}
    farm_performance = []
    
    if not uploaded_files:
        return {}, []
    
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Extract farm name from filename
        filename = uploaded_file.name
        if filename.endswith('_indices_timeseries.csv'):
            farm_name = filename.replace('_indices_timeseries.csv', '')
        else:
            farm_name = filename.replace('.csv', '')
        
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Handle different time formats
            if 'time' in df.columns:
                if df['time'].dtype == 'object':
                    try:
                        df['time'] = pd.to_datetime(df['time'])
                    except:
                        df['time'] = pd.to_datetime(df['time'], unit='ms')
                else:
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                df = df.set_index('time')
            
            # Ensure NDVI column exists
            if 'NDVI' not in df.columns:
                st.warning(f"No NDVI column found in {filename}. Skipping...")
                continue
            
            # Calculate KPIs
            peak_ndvi = df['NDVI'].max()
            avg_ndvi = df['NDVI'].mean()
            min_ndvi = df['NDVI'].min()
            std_ndvi = df['NDVI'].std()
            
            # Monthly aggregation
            df_monthly = df.resample('M').mean()
            
            all_farms_data[farm_name] = {
                'raw_data': df,
                'monthly_data': df_monthly,
                'kpis': {
                    'peak_ndvi': peak_ndvi,
                    'avg_ndvi': avg_ndvi,
                    'min_ndvi': min_ndvi,
                    'std_ndvi': std_ndvi,
                    'data_points': len(df)
                }
            }
            
            farm_performance.append({
                'farm_name': farm_name,
                'avg_ndvi': avg_ndvi,
                'peak_ndvi': peak_ndvi,
                'min_ndvi': min_ndvi,
                'volatility': std_ndvi
            })
            
        except Exception as e:
            st.warning(f"Error processing {filename}: {e}")
    
    progress_bar.empty()
    return all_farms_data, farm_performance

def create_seasonal_analysis(all_farms_data):
    """Create seasonal analysis"""
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

def main():
    st.markdown('<h1 class="main-header">🌴 Palm Farm Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("📁 Upload Your Farm Data")
    st.markdown("""
    **Instructions:**
    1. Upload your CSV files containing farm time-series data
    2. Files should have columns: 'time' and 'NDVI'
    3. You can upload multiple files at once
    4. Supported formats: CSV files
    """)
    
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload your farm time-series data files. Each file should contain 'time' and 'NDVI' columns."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            all_farms_data, farm_performance = process_uploaded_files(uploaded_files)
        
        if not all_farms_data:
            st.error("No valid data found in uploaded files. Please check your CSV format.")
            st.markdown("""
            **Expected CSV format:**
            ```
            time,NDVI
            1609459200000,0.75
            1609545600000,0.72
            ...
            ```
            """)
            return
        
        # Store data in session state
        st.session_state.farms_data = all_farms_data
        st.session_state.farm_performance = farm_performance
        
        st.success(f"✅ Successfully processed {len(all_farms_data)} farms!")
    
    # Check if data exists
    if 'farms_data' not in st.session_state or not st.session_state.farms_data:
        st.info("👆 Please upload your farm data files to get started")
        
        # Show sample data format
        st.subheader("📋 Sample Data Format")
        sample_data = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=12, freq='M'),
            'NDVI': np.random.uniform(0.3, 0.8, 12)
        })
        sample_data['time'] = sample_data['time'].astype(str)
        st.dataframe(sample_data)
        return
    
    all_farms_data = st.session_state.farms_data
    farm_performance = st.session_state.farm_performance
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "📈 Farm Comparison", "📅 Seasonal Analysis", "🔍 Individual Farm Details"])
    
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
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
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
            
            # Seasonal recommendations
            st.subheader("📈 Seasonal Management Recommendations")
            
            seasonal_range = seasonal_avg.max() - seasonal_avg.min()
            if seasonal_range > 0.2:
                st.warning(f"High seasonal variability detected (range: {seasonal_range:.3f}). Consider:")
                st.write("• Implementing seasonal irrigation adjustments")
                st.write("• Planning maintenance during low-growth periods")
                st.write("• Monitoring more closely during challenging months")
            else:
                st.success(f"Good seasonal stability (range: {seasonal_range:.3f}). Current management appears effective.")
    
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
                    csv_data = farm_data['raw_data'].to_csv()
                    st.download_button(
                        label=f"Download {selected_farm} Raw Data",
                        data=csv_data,
                        file_name=f"{selected_farm}_raw_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    monthly_csv = farm_data['monthly_data'].to_csv()
                    st.download_button(
                        label=f"Download {selected_farm} Monthly Summary",
                        data=monthly_csv,
                        file_name=f"{selected_farm}_monthly_summary.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*Palm Farm Analytics Dashboard | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Data from {len(st.session_state.get('farms_data', {}))} farms*"
    )

if __name__ == "__main__":
    main()
