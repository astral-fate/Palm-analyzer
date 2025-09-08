
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os

# --- Configuration ---
# IMPORTANT: Update this to the root directory containing all your farm folders
data_root_folder = '/content/drive/MyDrive/palm/data/'

# --- 1. Discover and Consolidate All Enriched Farm Data ---
def consolidate_farm_data(root_folder):
    """
    Finds all enriched data files, adds a farm_name column, and combines them.
    """
    # Use a flexible pattern to find the correct enriched files
    search_pattern = os.path.join(root_folder, '**', '*_s2_timeseries_enriched.csv')
    enriched_files = glob.glob(search_pattern, recursive=True)

    if not enriched_files:
        print("No enriched files found with the pattern '*_s2_timeseries_enriched.csv'.")
        print("Looking for older '*_enriched.csv' pattern...")
        search_pattern = os.path.join(root_folder, '**', '*_enriched.csv')
        enriched_files = glob.glob(search_pattern, recursive=True)

    if not enriched_files:
        print("ERROR: Could not find any enriched data files. Please check the folder and file names.")
        return pd.DataFrame()

    print(f"Found {len(enriched_files)} enriched farm data files. Consolidating...")

    all_farms_df_list = []
    for file in enriched_files:
        try:
            # Extract farm name from the folder path
            farm_name = os.path.basename(os.path.dirname(file))
            df = pd.read_csv(file)
            df['farm_name'] = farm_name # Add farm identifier
            all_farms_df_list.append(df)
            print(f"  - Loaded {len(df)} records for {farm_name}")
        except Exception as e:
            print(f"  - Could not load or process {file}: {e}")

    if not all_farms_df_list:
        print("ERROR: No data could be loaded.")
        return pd.DataFrame()

    consolidated_df = pd.concat(all_farms_df_list, ignore_index=True)
    consolidated_df['timestamp'] = pd.to_datetime(consolidated_df['timestamp'])
    print(f"\nConsolidation complete. Total records: {len(consolidated_df)}")
    return consolidated_df

# --- 2. Anomaly Detection Model ---
def classify_anomalies_for_all_farms(df):
    """
    Processes each farm's data individually to detect and classify anomalies.
    """
    all_anomalies = {}

    for farm_name, farm_data in df.groupby('farm_name'):
        print(f"\n--- Analyzing {farm_name} ---")

        # Set timestamp as index for resampling
        farm_data = farm_data.sort_values('timestamp').set_index('timestamp')

        # Preprocess: Resample weekly and calculate rate of change
        metrics = ['NDVI', 'NDWI', 'SAR_VV', 'SAR_VH']
        df_resampled = farm_data[metrics].resample('W').mean().interpolate(method='linear')
        df_change = df_resampled.diff().dropna()

        if df_change.empty:
            print("  - Not enough data to calculate changes. Skipping.")
            continue

        # Calculate dynamic thresholds based on this farm's specific variance
        rolling_std = df_change.rolling(window=12, min_periods=4).std()
        thresholds = {
            'NDVI': rolling_std['NDVI'] * 1.5,
            'NDWI': rolling_std['NDWI'] * 1.5,
            'SAR_VV': rolling_std['SAR_VV'] * 1.5
        }

        farm_anomalies = {}
        for date, row in df_change.iterrows():
            ndvi_change = row['NDVI']
            ndwi_change = row['NDWI']
            sar_vv_change = row['SAR_VV']

            # Get dynamic thresholds for the specific date
            ndvi_thresh = thresholds['NDVI'].get(date, 0.07) # Default fallback
            ndwi_thresh = thresholds['NDWI'].get(date, 0.07)
            sar_thresh = thresholds['SAR_VV'].get(date, 1.0)

            # Classification Logic
            if ndvi_change < -ndvi_thresh and sar_vv_change < -sar_thresh:
                farm_anomalies[date] = 'Harvest Event'
            elif ndvi_change < -ndvi_thresh and ndwi_change < -ndwi_thresh:
                farm_anomalies[date] = 'Potential Drought Stress'
            elif ndvi_change < -ndvi_thresh:
                farm_anomalies[date] = 'General Stress Event'

        print(f"  - Found {len(farm_anomalies)} potential anomalies.")
        all_anomalies[farm_name] = {'anomalies': farm_anomalies, 'data': df_resampled}

    return all_anomalies

# --- 3. Visualization ---
def visualize_farm_results(farm_name, analysis_results):
    """
    Plots the time series and classified anomalies for a specific farm.
    """
    if farm_name not in analysis_results:
        print(f"ERROR: No analysis results found for '{farm_name}'. Available farms: {list(analysis_results.keys())}")
        return

    farm_info = analysis_results[farm_name]
    original_data = farm_info['data']
    anomalies = farm_info['anomalies']

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Anomaly Detection for {farm_name}', fontsize=16)

    colors = {'Harvest Event': 'red', 'Potential Drought Stress': 'orange', 'General Stress Event': 'purple'}

    # Plot 1: Optical & SAR Data
    axes[0].plot(original_data.index, original_data['NDVI'], label='NDVI', color='green')
    axes[0].plot(original_data.index, original_data['NDWI'], label='NDWI', color='blue', linestyle=':')
    ax2 = axes[0].twinx() # Create a second y-axis for SAR
    ax2.plot(original_data.index, original_data['SAR_VV'], label='SAR_VV', color='black', linestyle='--')
    axes[0].set_ylabel('Optical Index Value')
    ax2.set_ylabel('SAR Backscatter (dB)')
    axes[0].set_title('Satellite Metrics Over Time')

    # Plot 2: NDVI Rate of Change
    df_change = original_data.diff()
    axes[1].plot(df_change.index, df_change['NDVI'], label='Weekly NDVI Change', color='darkgreen')
    axes[1].axhline(0, color='grey', linestyle=':', linewidth=1)
    axes[1].set_ylabel('Weekly Change')
    axes[1].set_title('NDVI Rate of Change')

    # Highlight anomalies on both plots
    for date, classification in anomalies.items():
        color = colors.get(classification, 'grey')
        for ax in [axes[0], axes[1]]:
            ax.axvline(x=date, color=color, linestyle='--', linewidth=1.5, label=f'{classification} ({date.date()})')

    # Create a single combined legend
    lines, labels = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels + labels2, lines + lines2))

    # Add anomaly labels to legend
    for c_label, c_color in colors.items():
        if any(c_label in str(l) for l in plt.gca().get_legend_handles_labels()[1]):
             unique_labels[c_label] = plt.Line2D([0], [0], color=c_color, linestyle='--')

    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Consolidate data
    master_df = consolidate_farm_data(data_root_folder)

    if not master_df.empty:
        # 2. Run analysis on all farms
        analysis_results = classify_anomalies_for_all_farms(master_df)

        # 3. Print a summary report
        print("\n\n--- MASTER ANOMALY REPORT ---")
        total_anomalies = 0
        for farm, results in analysis_results.items():
            num_anomalies = len(results['anomalies'])
            if num_anomalies > 0:
                print(f"\nFarm: {farm} ({num_anomalies} anomalies found)")
                for date, classification in sorted(results['anomalies'].items()):
                    print(f"  - {date.date()}: {classification}")
                total_anomalies += num_anomalies
        print(f"\n--- End of Report: {total_anomalies} total anomalies found across all farms. ---")

        # 4. Visualize a specific farm (you can change the name)
        if analysis_results:
            farm_to_visualize = list(analysis_results.keys())[0] # Visualize the first farm
            print(f"\n--- Visualizing results for a sample farm: {farm_to_visualize} ---")
            visualize_farm_results(farm_to_visualize, analysis_results)
