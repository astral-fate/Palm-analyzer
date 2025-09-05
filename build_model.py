import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

def engineer_yearly_features(df):
    """
    Analyzes time-series data to create yearly performance features for each farm.
    """
    print("Starting yearly feature engineering...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['day_of_year'] = df['timestamp'].dt.dayofyear

    yearly_features = []
    
    for farm in df['farm_name'].unique():
        for year in df[df['farm_name'] == farm]['year'].unique():
            farm_year_df = df[(df['farm_name'] == farm) & (df['year'] == year)]
            if farm_year_df.empty:
                continue

            peak_ndvi = farm_year_df['NDVI'].max()
            peak_day = farm_year_df.loc[farm_year_df['NDVI'].idxmax()]['day_of_year']
            min_ndvi_pre_peak = farm_year_df[farm_year_df['day_of_year'] < peak_day]['NDVI'].min()
            start_day_series = farm_year_df[farm_year_df['NDVI'] == min_ndvi_pre_peak]['day_of_year']
            start_day = start_day_series.min() if not start_day_series.empty else 0
            
            season_duration = peak_day - start_day if not pd.isna(start_day) and start_day > 0 else 0
            summer_df = farm_year_df[farm_year_df['timestamp'].dt.month.isin([6, 7, 8])]
            avg_ndwi_stress = summer_df['NDWI'].mean() if not summer_df.empty else df['NDWI'].mean()
            seasonal_integral = farm_year_df['NDVI'].sum()
            ndvi_std_dev = farm_year_df['NDVI'].std()

            yearly_features.append({
                'farm_name': farm, 'year': year, 'peak_ndvi': peak_ndvi,
                'peak_day': peak_day, 'season_duration': season_duration,
                'avg_ndwi_stress': avg_ndwi_stress, 'seasonal_integral': seasonal_integral,
                'ndvi_std_dev': ndvi_std_dev
            })
            
    feature_df = pd.DataFrame(yearly_features).fillna(0)
    print("Feature engineering complete.")
    return feature_df

def cluster_farm_performance(feature_df, model_path):
    """
    Uses K-Means clustering to create a 'Performance Score' and saves the models.
    """
    print("Clustering farm performance...")
    features_for_clustering = feature_df.drop(columns=['farm_name', 'year'])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    feature_df['performance_cluster'] = kmeans.labels_
    
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=features_for_clustering.columns)
    
    tier_map = {
        centroid_df['peak_ndvi'].idxmax(): 'Premium Tier',
        centroid_df['peak_ndvi'].idxmin(): 'Economy Tier'
    }
    remaining_cluster = [c for c in [0, 1, 2] if c not in tier_map.keys()][0]
    tier_map[remaining_cluster] = 'Standard Tier'

    feature_df['performance_score'] = feature_df['performance_cluster'].map(tier_map)
    
    # Create the model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save the models and tier map inside the "Model" folder
    joblib.dump(kmeans, os.path.join(model_path, 'kmeans_model.joblib'))
    joblib.dump(scaler, os.path.join(model_path, 'scaler.joblib'))
    joblib.dump(tier_map, os.path.join(model_path, 'tier_map.joblib'))
    
    print(f"Clustering complete. Models saved to '{model_path}'.")
    return feature_df

def load_and_consolidate_data(folder_path):
    """Loads and consolidates historical data from a directory of farm folders."""
    all_farm_data = []
    if not os.path.isdir(folder_path):
        print(f"Error: The directory '{folder_path}' was not found.")
        return pd.DataFrame()

    farm_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for farm_name in farm_folders:
        farm_dir = os.path.join(folder_path, farm_name)
        try:
            target_file = next(f for f in os.listdir(farm_dir) if f.endswith('_historical_2015_2025.csv'))
            file_path = os.path.join(farm_dir, target_file)
            df_farm = pd.read_csv(file_path)
            df_farm['farm_name'] = farm_name  # Use folder name for consistency
            all_farm_data.append(df_farm)
        except StopIteration:
            print(f"Warning: No historical CSV file found in folder: {farm_name}")
    
    if not all_farm_data:
        print("No valid data could be loaded.")
        return pd.DataFrame()
        
    return pd.concat(all_farm_data, ignore_index=True)

if __name__ == "__main__":
    # Define the paths
    data_folder = "Data"
    model_folder = "Model"

    # Load the raw data
    df = load_and_consolidate_data(data_folder)
    
    if not df.empty:
        # Engineer the features
        yearly_data = engineer_yearly_features(df)
        
        # Cluster and save the models
        clustered_data = cluster_farm_performance(yearly_data, model_folder)
        
        # Save the final processed data into the Model folder
        output_path = os.path.join(model_folder, "farm_yearly_performance.csv")
        clustered_data.to_csv(output_path, index=False)
        
        print(f"\nSuccessfully created '{output_path}' with performance scores.")
