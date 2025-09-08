# FILE: 2_final_performance_report.py

import pandas as pd
import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration ---
data_root_folder = '/content/drive/MyDrive/palm/data/'

def load_and_prepare_data(root_folder):
    """Loads and consolidates all the final enriched farm data."""
    search_pattern = os.path.join(root_folder, '**', '*_enriched_data.csv')
    enriched_files = glob.glob(search_pattern, recursive=True)
    if not enriched_files:
        print("ERROR: No enriched data files found. Please run the SAR enrichment script first.")
        return pd.DataFrame()
    df_list = []
    for file in enriched_files:
        farm_name = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        df['farm_name'] = farm_name
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Consolidated data from {len(enriched_files)} farms. Total records: {len(df)}")
    return df

def calculate_performance_tiers(df):
    """Calculates performance KPIs and uses K-Means clustering to assign tiers."""
    print("\nCalculating farm performance tiers...")
    kpi_df = df.groupby('farm_name').agg(
        mean_ndvi=('NDVI', 'mean'), mean_evi=('EVI', 'mean'), std_ndvi=('NDVI', 'std')
    ).reset_index().dropna()
    features = kpi_df[['mean_ndvi', 'mean_evi', 'std_ndvi']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kpi_df['cluster'] = kmeans.fit_predict(scaled_features)
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['mean_ndvi', 'mean_evi', 'std_ndvi'])
    sorted_clusters = cluster_centers.sort_values(by='mean_ndvi', ascending=False).index
    tier_map = {sorted_clusters[0]: 'Tier 1 (High)', sorted_clusters[1]: 'Tier 2 (Medium)', sorted_clusters[2]: 'Tier 3 (Low)'}
    kpi_df['performance_tier'] = kpi_df['cluster'].map(tier_map)
    print("Performance tiers calculated successfully.")
    return kpi_df[['farm_name', 'performance_tier', 'mean_ndvi', 'mean_evi', 'std_ndvi']]

def evaluate_farm_forecasts(df):
    """Trains and evaluates an NDVI forecasting model for each farm."""
    print("\nEvaluating forecasting model for each farm...")
    forecast_metrics = []
    for farm_name, farm_data in df.groupby('farm_name'):
        farm_data = farm_data.set_index('timestamp').sort_index().dropna(subset=['NDVI', 'EVI', 'NDWI'])
        farm_data['day_of_year'] = farm_data.index.dayofyear
        X = farm_data[['day_of_year', 'EVI', 'NDWI']]
        y = farm_data['NDVI']
        if len(farm_data) < 50: continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        forecast_metrics.append({'farm_name': farm_name, 'forecast_r2': r2, 'forecast_mae': mae})
        print(f"  - {farm_name}: R² = {r2:.3f}, MAE = {mae:.3f}")
    return pd.DataFrame(forecast_metrics)

if __name__ == '__main__':
    master_df = load_and_prepare_data(data_root_folder)
    if not master_df.empty:
        performance_df = calculate_performance_tiers(master_df)
        forecast_df = evaluate_farm_forecasts(master_df)
        if not (performance_df.empty or forecast_df.empty):
            final_report_df = pd.merge(performance_df, forecast_df, on='farm_name')
            final_report_df = final_report_df.sort_values(by=['performance_tier', 'forecast_r2'], ascending=[True, False])
            
            print("\n\n" + "="*80)
            print(" " * 25 + "FINAL FARM PERFORMANCE REPORT")
            print("="*80)
            print("Ranks farms by performance tier (overall quality) and forecast R² (predictability).\n")
            for col in ['mean_ndvi', 'mean_evi', 'std_ndvi', 'forecast_r2', 'forecast_mae']:
                final_report_df[col] = final_report_df[col].astype(float).map('{:.3f}'.format)
            print(final_report_df.to_string(index=False))
            print("="*80)
