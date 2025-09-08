import pandas as pd
import glob
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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


# --- 3. Train the Enriched Clustering Model ---
def train_and_evaluate_clustering(df):
    """
    Calculates performance KPIs (including EVI), trains the K-Means model,
    and assigns final performance tiers to each farm.
    """
    print("\n--- Training Enriched Clustering Model ---")
    
    # a) Calculate Key Performance Indicators (KPIs) for each farm
    # We use mean NDVI and EVI for vigor, and std NDVI for stability.
    kpi_df = df.groupby('farm_name').agg(
        mean_ndvi=('NDVI', 'mean'),
        mean_evi=('EVI', 'mean'),
        std_ndvi=('NDVI', 'std')
    ).reset_index().dropna()

    print("Calculated KPIs for each farm:")
    print(kpi_df.head())

    # b) Scale the features to ensure fair comparison
    features_to_cluster = kpi_df[['mean_ndvi', 'mean_evi', 'std_ndvi']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_to_cluster)

    # c) Train the K-Means model to find 3 distinct performance groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kpi_df['cluster'] = kmeans.fit_predict(scaled_features)
    print("\n✓ Model training complete.")

    # d) Map cluster labels to human-readable tiers (High, Medium, Low)
    # We determine which cluster is "best" by looking at the average NDVI of its members.
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['mean_ndvi', 'mean_evi', 'std_ndvi'])
    sorted_clusters = cluster_centers.sort_values(by='mean_ndvi', ascending=False).index
    
    tier_map = {
        sorted_clusters[0]: 'Tier 1 (High Performance)',
        sorted_clusters[1]: 'Tier 2 (Medium Performance)',
        sorted_clusters[2]: 'Tier 3 (Low Performance)'
    }
    kpi_df['Performance Tier'] = kpi_df['cluster'].map(tier_map)

    # e) Save the final, trained models for use in the application
    joblib.dump(scaler, os.path.join(model_output_folder, 'scaler.joblib'))
    joblib.dump(kmeans, os.path.join(model_output_folder, 'kmeans_model.joblib'))
    print(f"✓ Final models ('scaler.joblib', 'kmeans_model.joblib') saved to: {model_output_folder}")

    return kpi_df.sort_values(by='Performance Tier')


# --- 4. Main Execution ---
if __name__ == '__main__':
    # Load the accurate, enriched data
    master_df = load_and_prepare_data(data_root_folder)
    
    if not master_df.empty:
        # Run the clustering experiment
        final_report = train_and_evaluate_clustering(master_df)

        # Display the final results
        print("\n\n" + "="*80)
        print(" " * 20 + "FINAL ENRICHED CLUSTERING MODEL RESULTS")
        print("="*80)
        print("Farms have been clustered into performance tiers based on their long-term\nNDVI, EVI, and stability metrics.\n")
        
        # Format for readability
        report_display = final_report[['farm_name', 'Performance Tier', 'mean_ndvi', 'mean_evi', 'std_ndvi']]
        for col in ['mean_ndvi', 'mean_evi', 'std_ndvi']:
             report_display[col] = report_display[col].map('{:.3f}'.format)
        
        print(report_display.to_string(index=False))
        print("="*80)
