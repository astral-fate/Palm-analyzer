# FILE: build_and_save_models.py

import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# --- Configuration ---
data_root_folder = '/content/drive/MyDrive/palm/data/'
model_output_folder = '/content/drive/MyDrive/palm/data/Model/'

# Create the model directory if it doesn't exist
os.makedirs(model_output_folder, exist_ok=True)

# --- 1. Load Data ---
def load_data(root_folder):
    search_pattern = os.path.join(root_folder, '**', '*_enriched_data.csv')
    files = glob.glob(search_pattern, recursive=True)
    if not files: raise FileNotFoundError("No enriched data files found.")
    df_list = []
    for file in files:
        farm_name = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        df['farm_name'] = farm_name
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- 2. Train and Save Performance Tiering Model ---
def build_performance_model(df):
    print("\n--- Building and Saving Performance Tiering Model ---")
    kpi_df = df.groupby('farm_name').agg(
        mean_ndvi=('NDVI', 'mean'), mean_evi=('EVI', 'mean'), std_ndvi=('NDVI', 'std')
    ).reset_index().dropna()
    
    features = kpi_df[['mean_ndvi', 'mean_evi', 'std_ndvi']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kpi_df['cluster'] = kmeans.fit_predict(scaled_features)

    # Save the scaler and model
    joblib.dump(scaler, os.path.join(model_output_folder, 'scaler.joblib'))
    joblib.dump(kmeans, os.path.join(model_output_folder, 'kmeans_model.joblib'))
    print("  ✓ Scaler and K-Means model saved.")
    return kpi_df

# --- 3. Train and Save Forecasting Models ---
def build_forecasting_models(df):
    print("\n--- Building and Saving Forecasting Models for Each Farm ---")
    forecasting_models = {}
    for farm_name, farm_data in df.groupby('farm_name'):
        farm_data = farm_data.set_index('timestamp').sort_index().dropna(subset=['NDVI', 'EVI', 'NDWI'])
        farm_data['day_of_year'] = farm_data.index.dayofyear
        
        X = farm_data[['day_of_year', 'EVI', 'NDWI']]
        y = farm_data['NDVI']
        
        if len(farm_data) < 50: continue
            
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        forecasting_models[farm_name] = model
        print(f"  - Trained and stored model for {farm_name}")
        
    # Save all farm models in a single file
    joblib.dump(forecasting_models, os.path.join(model_output_folder, 'forecasting_models.joblib'))
    print("  ✓ All forecasting models saved.")

# --- Main Execution ---
if __name__ == '__main__':
    master_df = load_data(data_root_folder)
    build_performance_model(master_df)
    build_forecasting_models(master_df)
    # Save the final consolidated data for the app to use
    master_df.to_csv(os.path.join(model_output_folder, 'consolidated_farm_data.csv'), index=False)
    print("\n--- Model building complete. All models and data are saved in the 'Model' directory. ---")
