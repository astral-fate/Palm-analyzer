import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def engineer_yearly_features(df):
    """
    Analyzes time-series data to create yearly performance features for each farm.
    """
    print("Starting yearly feature engineering...")
    
    # Ensure timestamp is a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['day_of_year'] = df['timestamp'].dt.dayofyear

    yearly_features = []
    
    for farm in df['farm_name'].unique():
        for year in df[df['farm_name'] == farm]['year'].unique():
            farm_year_df = df[(df['farm_name'] == farm) & (df['year'] == year)]
            
            if farm_year_df.empty:
                continue

            # 1. Peak Health
            peak_ndvi = farm_year_df['NDVI'].max()
            peak_day = farm_year_df.loc[farm_year_df['NDVI'].idxmax()]['day_of_year']

            # 2. Growth Start and Duration
            min_ndvi_pre_peak = farm_year_df[farm_year_df['day_of_year'] < peak_day]['NDVI'].min()
            start_day = farm_year_df[farm_year_df['NDVI'] == min_ndvi_pre_peak]['day_of_year'].min()
            
            season_duration = peak_day - start_day if not pd.isna(start_day) else 0

            # 3. Water Stress (avg NDWI in summer)
            summer_df = farm_year_df[farm_year_df['timestamp'].dt.month.isin([6, 7, 8])]
            avg_ndwi_stress = summer_df['NDWI'].mean() if not summer_df.empty else df['NDWI'].mean()

            # 4. Overall Vigor (Area under NDVI curve)
            seasonal_integral = farm_year_df['NDVI'].sum()

            # 5. Stability
            ndvi_std_dev = farm_year_df['NDVI'].std()

            yearly_features.append({
                'farm_name': farm,
                'year': year,
                'peak_ndvi': peak_ndvi,
                'peak_day': peak_day,
                'season_duration': season_duration,
                'avg_ndwi_stress': avg_ndwi_stress,
                'seasonal_integral': seasonal_integral,
                'ndvi_std_dev': ndvi_std_dev
            })
            
    feature_df = pd.DataFrame(yearly_features).fillna(0)
    print("Feature engineering complete.")
    return feature_df


def cluster_farm_performance(feature_df):
    """
    Uses K-Means clustering to create a 'Performance Score' for each farm-year.
    """
    print("Clustering farm performance...")
    
    features_for_clustering = feature_df.drop(columns=['farm_name', 'year'])
    
    # Scale features for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)
    
    # Train K-Means model
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    # Add cluster labels to the dataframe
    feature_df['performance_cluster'] = kmeans.labels_

    # Analyze centroids to name clusters
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=features_for_clustering.columns)
    
    # Name clusters based on peak_ndvi (higher is better)
    tier_map = {
        centroid_df['peak_ndvi'].idxmax(): 'Premium Tier',
        centroid_df['peak_ndvi'].idxmin(): 'Economy Tier'
    }
    # The remaining cluster is 'Standard Tier'
    remaining_cluster = [c for c in [0, 1, 2] if c not in tier_map.keys()][0]
    tier_map[remaining_cluster] = 'Standard Tier'

    feature_df['performance_score'] = feature_df['performance_cluster'].map(tier_map)
    
    # Save the scaler and model for the app
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(kmeans, 'kmeans_model.joblib')
    joblib.dump(tier_map, 'tier_map.joblib')
    
    print("Clustering complete. Models saved.")
    print("\nCluster Centroids (Characteristics):")
    print(centroid_df)
    print("\nTier Mapping:")
    print(tier_map)
    
    return feature_df


if __name__ == "__main__":
    df = pd.read_csv("/content/drive/MyDrive/palm/data/consolidated_historical_2015_2025.csv")
    yearly_data = engineer_yearly_features(df)
    clustered_data = cluster_farm_performance(yearly_data)
    
    # Save the final processed data for the app
    clustered_data.to_csv("farm_yearly_performance.csv", index=False)
    
    print("\nSuccessfully created 'farm_yearly_performance.csv' with performance scores.")
