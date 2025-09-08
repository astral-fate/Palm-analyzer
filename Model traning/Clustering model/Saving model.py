import pandas as pd
import glob
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# --- 1. Configuration ---
# Define the root folder where your farm data is located
data_root_folder = '/content/drive/MyDrive/palm/data/'
# Define where the final, trained models will be saved
model_output_folder = '/content/drive/MyDrive/palm/data/Model/'
os.makedirs(model_output_folder, exist_ok=True)


# --- 2. Load and Consolidate Final Enriched Data ---
def load_and_prepare_data(root_folder):
    """
    Loads and consolidates all the final '_enriched_data.csv' files from each farm folder.
    """
    print("--- Loading Final Enriched Data ---")
    search_pattern = os.path.join(root_folder, '**', '*_enriched_data.csv')
    enriched_files = glob.glob(search_pattern, recursive=True)
    
    if not enriched_files:
        raise FileNotFoundError("ERROR: No '_enriched_data.csv' files found. Please ensure the data collection and SAR enrichment scripts have been run successfully.")

    df_list = []
    for file in enriched_files:
        try:
            farm_name = os.path.basename(os.path.dirname(file))
            df = pd.read_csv(file)
            df['farm_name'] = farm_name
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}. Error: {e}")
            
    master_df = pd.concat(df_list, ignore_index=True)
    master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])
    
    print(f"✓ Consolidated data from {len(enriched_files)} farms. Total records: {len(master_df)}")
    return master_df


# --- 3. Train and Save Clustering Model ---
def train_clustering_model(df):
    """
    Trains the K-Means clustering model and saves it along with the scaler.
    """
    print("\n--- Training and Saving Enriched Clustering Model ---")
    
    kpi_df = df.groupby('farm_name').agg(
        mean_ndvi=('NDVI', 'mean'),
        mean_evi=('EVI', 'mean'),
        std_ndvi=('NDVI', 'std')
    ).reset_index().dropna()

    features_to_cluster = kpi_df[['mean_ndvi', 'mean_evi', 'std_ndvi']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_to_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    print("✓ Clustering model training complete.")

    # Save the final, trained models for use in the application
    joblib.dump(scaler, os.path.join(model_output_folder, 'scaler.joblib'))
    joblib.dump(kmeans, os.path.join(model_output_folder, 'kmeans_model.joblib'))
    print(f"✓ Models ('scaler.joblib', 'kmeans_model.joblib') saved to: {model_output_folder}")

    return kpi_df


# --- 4. Train and Save Forecasting Models ---
def train_forecasting_models(df):
    """
    Trains a specific forecasting model for each farm and saves them all in one file.
    """
    print("\n--- Training and Saving Forecasting Models ---")
    forecasting_models = {}
    for farm_name, farm_data in df.groupby('farm_name'):
        farm_data = farm_data.set_index('timestamp').sort_index().dropna(subset=['NDVI', 'EVI', 'NDWI'])
        farm_data['day_of_year'] = farm_data.index.dayofyear
        
        X = farm_data[['day_of_year', 'EVI', 'NDWI']]
        y = farm_data['NDVI']
        
        if len(farm_data) < 50:
            print(f"  - Skipping {farm_name} due to insufficient data.")
            continue
            
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        forecasting_models[farm_name] = model
        print(f"  - Trained model for {farm_name}")

    joblib.dump(forecasting_models, os.path.join(model_output_folder, 'forecasting_models.joblib'))
    print(f"✓ All forecasting models saved to: {os.path.join(model_output_folder, 'forecasting_models.joblib')}")


# --- 5. Main Execution ---
if __name__ == '__main__':
    # Load the accurate, enriched data
    master_df = load_and_prepare_data(data_root_folder)
    
    if not master_df.empty:
        # Train and save the clustering model
        train_clustering_model(master_df)
        
        # Train and save all forecasting models
        train_forecasting_models(master_df)

        # Save the consolidated data file needed for the app
        master_df.to_csv(os.path.join(model_output_folder, 'consolidated_farm_data.csv'), index=False)
        print(f"\n✓ Consolidated data saved to the 'Model' directory.")
        
        print("\n\n" + "="*80)
        print(" " * 25 + "MODEL BUILDING COMPLETE")
        print("="*80)
        print("All models and data are now saved and ready for use in your Gradio application.")
        print("="*80)
