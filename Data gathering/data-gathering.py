import ee
import geemap
import pandas as pd
import os
import json
from google.colab import drive
from ipywidgets import Button, Layout, VBox, HBox, Text, Output
from IPython.display import display, clear_output
from datetime import datetime

# Mount your Google Drive
drive.mount('/content/drive')

# Step 2: Authenticate and initialize Earth Engine
try:
    ee.Initialize(project='dotted-medley-467420-k3')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='dotted-medley-467420-k3')
    
# Step 3: Set up file paths and data structures
gdrive_folder = '/content/drive/MyDrive/palm/data/'
farm_list_csv = os.path.join(gdrive_folder, 'my_farm_list.csv')

if not os.path.exists(gdrive_folder):
    os.makedirs(gdrive_folder)

farm_collection = []

# Step 4: Create Interactive Map and Widgets
m = geemap.Map(center=[24.45, 39.62], zoom=12, layout=Layout(height='500px'))
m.add_basemap('SATELLITE')

farm_name_input = Text(value='', placeholder='Enter farm name here', description='Farm Name:', disabled=False)
add_button = Button(description="Add Farm to List", button_style='primary')
process_button = Button(description="Process and Save All Farms to Drive", button_style='success')
output_widget = Output()

# Simple but robust data collection function
def enhanced_data_collection(farm_aoi, farm_name):
    """
    Collect data using the most basic, reliable approach
    """
    print(f"  - Collecting Sentinel-2 vegetation indices...")
    
    try:
        # Use the basic approach that we know works
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(farm_aoi)
                      .filterDate('2022-01-01', '2024-12-31')
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                      .sort('system:time_start'))
        
        print(f"  - Found {collection.size().getInfo()} potential images")
        
        if collection.size().getInfo() == 0:
            return pd.DataFrame()
        
        # Simple index calculation
        def calculate_indices(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
            savi = image.expression('1.5 * (NIR - RED) / (NIR + RED + 0.5)', 
                                    {'NIR': image.select('B8'), 'RED': image.select('B4')}).rename('SAVI')
            return image.addBands([ndvi, ndwi, savi]).copyProperties(image, ['system:time_start'])

        def get_mean_values(image):
            mean_dict = image.reduceRegion(
                reducer=ee.Reducer.mean(), 
                geometry=farm_aoi, 
                scale=20, 
                maxPixels=1e8
            )
            
            date = ee.Date(image.get('system:time_start'))
            
            return ee.Feature(None, {
                'time': image.get('system:time_start'),
                'year': date.get('year'),
                'month': date.get('month'),
                'day': date.get('day'),
                'NDVI': mean_dict.get('NDVI'),
                'NDWI': mean_dict.get('NDWI'),
                'SAVI': mean_dict.get('SAVI'),
                'cloud_percent': image.get('CLOUDY_PIXEL_PERCENTAGE')
            })

        indices_collection = collection.map(calculate_indices)
        mean_fc = ee.FeatureCollection(indices_collection.map(get_mean_values))
        
        # Filter only for non-null NDVI
        mean_fc = mean_fc.filter(ee.Filter.notNull(['NDVI']))
        
        print(f"  - After filtering: {mean_fc.size().getInfo()} valid observations")
        
        if mean_fc.size().getInfo() == 0:
            return pd.DataFrame()
        
        # Convert to pandas - this is where the error occurs
        try:
            df = geemap.ee_to_df(mean_fc)
        except Exception as e:
            print(f"  - geemap.ee_to_df failed: {e}")
            print(f"  - Trying alternative conversion method...")
            
            # Alternative: get data as a list and convert manually
            try:
                # Get first few features to test
                feature_list = mean_fc.limit(10).getInfo()
                if 'features' in feature_list and len(feature_list['features']) > 0:
                    print(f"  - Successfully retrieved sample data, now getting full dataset...")
                    
                    # Get all data
                    full_data = mean_fc.getInfo()
                    
                    # Convert to DataFrame manually
                    data_rows = []
                    for feature in full_data['features']:
                        if feature['properties']:
                            data_rows.append(feature['properties'])
                    
                    df = pd.DataFrame(data_rows)
                else:
                    print(f"  - No features found in the collection")
                    return pd.DataFrame()
                    
            except Exception as e2:
                print(f"  - Alternative method also failed: {e2}")
                return pd.DataFrame()
        
        if not df.empty:
            # Convert timestamp to proper datetime
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            
            # Sort by date
            df = df.sort_values('timestamp')
            
            # Add time features
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            df['quarter'] = df['timestamp'].dt.quarter
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            
            # Add season
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            print(f"  - Successfully collected {len(df)} observations")
            print(f"  - Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
            
        return df
        
    except Exception as e:
        print(f"  - Major error in data collection: {e}")
        import traceback
        print(f"  - Full traceback: {traceback.format_exc()}")
        return pd.DataFrame()

# Enhanced processing function
def process_all_farms_enhanced(b):
    with output_widget:
        clear_output(wait=True)
        if not farm_collection:
            print("No farms have been added to the list yet.")
            return
            
        print(f"Starting enhanced data collection for {len(farm_collection)} farms...")
        all_farms_data = []
        
        for farm in farm_collection:
            farm_name = farm['name']
            
            try:
                # Better geometry handling
                if 'geojson' in farm:
                    # Convert geojson to ee.Geometry properly
                    geojson = farm['geojson']
                    if isinstance(geojson, dict):
                        if 'geometry' in geojson:
                            farm_aoi = ee.Geometry(geojson['geometry'])
                        else:
                            farm_aoi = ee.Geometry(geojson)
                    else:
                        # If it's already a geometry object
                        farm_aoi = ee.Geometry(geojson)
                else:
                    print(f"  ✗ No geometry found for {farm_name}")
                    continue
                    
                farm_folder = os.path.join(gdrive_folder, farm_name)
                if not os.path.exists(farm_folder):
                    os.makedirs(farm_folder)
                    
                print(f"\nProcessing: {farm_name}")
                
                # Get comprehensive data
                df = enhanced_data_collection(farm_aoi, farm_name)
                
                if not df.empty:
                    # Add farm identifier
                    df['farm_id'] = farm_name
                    df['farm_name'] = farm_name
                    
                    # Save individual farm data
                    timeseries_filename = os.path.join(farm_folder, f'{farm_name}_enhanced_timeseries.csv')
                    df.to_csv(timeseries_filename, index=False)
                    print(f"  ✓ Saved enhanced time-series to CSV ({len(df)} records)")
                    
                    # Add to consolidated dataset
                    all_farms_data.append(df)
                    
                    # Save summary statistics
                    summary_stats = {
                        'farm_name': farm_name,
                        'total_observations': len(df),
                        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                        'avg_ndvi': float(df['NDVI'].mean()),
                        'ndvi_range': f"{df['NDVI'].min():.3f} to {df['NDVI'].max():.3f}",
                        'data_quality_score': float((1 - df['cloud_percent'].mean()/100) * 100)
                    }
                    
                    with open(os.path.join(farm_folder, f'{farm_name}_summary.json'), 'w') as f:
                        json.dump(summary_stats, f, indent=2, default=str)
                    
                else:
                    print(f"  ✗ No data found for {farm_name}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {farm_name}: {e}")
                # Print more detailed error info for debugging
                import traceback
                print(f"  Debug info: {traceback.format_exc()}")
        
        # Create consolidated dataset for ML
        if all_farms_data:
            print(f"\nCreating consolidated dataset...")
            consolidated_df = pd.concat(all_farms_data, ignore_index=True)
            
            # Sort by farm and date
            consolidated_df = consolidated_df.sort_values(['farm_id', 'timestamp'])
            
            # Save consolidated dataset
            consolidated_filename = os.path.join(gdrive_folder, 'consolidated_enhanced_farm_data.csv')
            consolidated_df.to_csv(consolidated_filename, index=False)
            
            print(f"✓ Consolidated dataset saved: {len(consolidated_df)} total records")
            print(f"✓ Farms included: {consolidated_df['farm_id'].nunique()}")
            print(f"✓ Date range: {consolidated_df['timestamp'].min()} to {consolidated_df['timestamp'].max()}")
            print(f"✓ Features available: {len([col for col in consolidated_df.columns if col not in ['farm_id', 'farm_name', 'timestamp', 'date', 'date_string']])}")
            
            # Create ML-ready dataset with proper time indexing
            ml_ready_df = consolidated_df.copy()
            ml_ready_df['time'] = ml_ready_df['timestamp'].astype('int64') // 10**6  # Convert to milliseconds
            
            # Reorder columns for ML pipeline
            time_cols = ['time', 'farm_id', 'farm_name']
            feature_cols = ['NDVI', 'NDWI', 'SAVI', 'EVI', 'MSI', 'CHL_RED_EDGE', 'BSI']
            metadata_cols = ['year', 'month', 'day', 'day_of_year', 'week_of_year', 'quarter', 'season']
            quality_cols = ['NDVI_std', 'cloud_percent']
            
            ml_columns = time_cols + feature_cols + metadata_cols + quality_cols
            available_columns = [col for col in ml_columns if col in ml_ready_df.columns]
            
            ml_dataset = ml_ready_df[available_columns]
            ml_filename = os.path.join(gdrive_folder, 'ml_ready_farm_data.csv')
            ml_dataset.to_csv(ml_filename, index=False)
            
            print(f"✓ ML-ready dataset saved: {ml_filename}")
        else:
            print("✗ No data collected from any farms")
        
        print(f"\n🎉 Enhanced data collection completed!")
        print(f"📁 Check your Google Drive folder: '{gdrive_folder}'")
        if all_farms_data:
            print(f"\n📊 Files created:")
            print(f"  • Individual farm data: [farm_name]_enhanced_timeseries.csv")
            print(f"  • Consolidated dataset: consolidated_enhanced_farm_data.csv")
            print(f"  • ML-ready dataset: ml_ready_farm_data.csv")

# Step 5: Define Functions for Buttons (keeping original simple version too)
def add_farm_to_list(b):
    with output_widget:
        if not m.user_roi:
            print("❌ Error: Please draw a boundary on the map first.")
            return
        if not farm_name_input.value:
            print("❌ Error: Please enter a name for the farm.")
            return
        
        farm_name = farm_name_input.value
        farm_geojson = m.user_roi
        farm_collection.append({'name': farm_name, 'geojson': farm_geojson})
        
        print(f"✅ Farm '{farm_name}' added to the list. Total farms to process: {len(farm_collection)}")
        farm_name_input.value = ''
        m.remove_drawn_features()

def process_all_farms(b):
    """Original simple processing function"""
    with output_widget:
        clear_output(wait=True)
        if not farm_collection:
            print("No farms have been added to the list yet.")
            return
            
        print(f"Starting data collection for {len(farm_collection)} farms...")
        
        for farm in farm_collection:
            farm_name = farm['name']
            farm_aoi = ee.Geometry(farm['geojson'])
            farm_folder = os.path.join(gdrive_folder, farm_name)
            if not os.path.exists(farm_folder):
                os.makedirs(farm_folder)
                
            print(f"\nProcessing: {farm_name}")

            # --- GEE Analysis ---
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(farm_aoi)
                          .filterDate('2020-01-01', '2024-12-31')  # Updated date range
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
            
            # --- Server-side calculation for all indices ---
            def calculate_indices(image):
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
                savi = image.expression('1.5 * (NIR - RED) / (NIR + RED + 0.5)', 
                                        {'NIR': image.select('B8'), 'RED': image.select('B4')}).rename('SAVI')
                return image.addBands([ndvi, ndwi, savi]).copyProperties(image, ['system:time_start'])

            def get_mean_values(image):
                mean_dict = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=farm_aoi, scale=10, maxPixels=1e9)
                
                # Extract date information for better time handling
                date = ee.Date(image.get('system:time_start'))
                
                return ee.Feature(None, {
                    'time': image.get('system:time_start'),
                    'date_string': date.format('YYYY-MM-dd'),
                    'NDVI': mean_dict.get('NDVI'),
                    'NDWI': mean_dict.get('NDWI'),
                    'SAVI': mean_dict.get('SAVI')
                })

            indices_collection = collection.map(calculate_indices)
            mean_fc = ee.FeatureCollection(indices_collection.map(get_mean_values)).filter(ee.Filter.notNull(['NDVI', 'NDWI', 'SAVI']))
            
            df = geemap.ee_to_df(mean_fc)
            
            if not df.empty:
                # Fix timestamp conversion
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df['date'] = pd.to_datetime(df['date_string'])
                df = df.sort_values('time')
                
                # Save the combined time-series data to a CSV file
                timeseries_filename = os.path.join(farm_folder, f'{farm_name}_indices_timeseries.csv')
                df.to_csv(timeseries_filename, index=False)
                print(f"  - Saved NDVI, NDWI, and SAVI time-series to CSV.")
            else:
                print(f"  - No data found for {farm_name}.")
        
        print("\n\n✅ Data collection process finished for all farms.")
        print(f"Check your Google Drive folder: '{gdrive_folder}' for the outputs.")

# Create enhanced button
enhanced_button = Button(description="Enhanced Data Collection for ML", button_style='warning')
enhanced_button.on_click(process_all_farms_enhanced)

# Link functions to buttons
add_button.on_click(add_farm_to_list)
process_button.on_click(process_all_farms)

# Step 6: Display the Interactive Tool
ui = VBox([
    m,
    HBox([farm_name_input, add_button]),
    process_button,
    enhanced_button,  # Add the enhanced option
    output_widget
])
display(ui)
