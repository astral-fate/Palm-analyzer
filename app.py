import ee
import geemap
import pandas as pd
import os
import json
from google.colab import drive
from ipywidgets import Button, Layout, VBox, HBox, Text, Output, Dropdown, IntSlider, Checkbox
from IPython.display import display, clear_output
from datetime import datetime, timedelta
import time

# Mount your Google Drive
drive.mount('/content/drive')

# Authenticate and initialize Earth Engine
try:
    ee.Initialize(project='dotted-medley-467420-k3')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='dotted-medley-467420-k3')

# Set up file paths and data structures
gdrive_folder = '/content/drive/MyDrive/palm/data/'
farm_list_csv = os.path.join(gdrive_folder, 'my_farm_list.csv')

if not os.path.exists(gdrive_folder):
    os.makedirs(gdrive_folder)

farm_collection = []

# Enhanced configuration widgets
print("🌴 Enhanced Palm Farm Data Collection System")
print("=" * 50)

# Create Interactive Map and Widgets
m = geemap.Map(center=[24.45, 39.62], zoom=12, layout=Layout(height='500px'))
m.add_basemap('SATELLITE')

# Configuration widgets
farm_name_input = Text(value='', placeholder='Enter farm name here', description='Farm Name:', disabled=False)

date_start_input = Text(value='2018-01-01', placeholder='YYYY-MM-DD', description='Start Date:', disabled=False)
date_end_input = Text(value='2024-12-31', placeholder='YYYY-MM-DD', description='End Date:', disabled=False)

cloud_threshold = IntSlider(value=20, min=5, max=50, step=5, description='Max Cloud %:', disabled=False)

collection_frequency = Dropdown(
    options=[('All Available', 'all'), ('Monthly', 'monthly'), ('Weekly', 'weekly')],
    value='all',
    description='Frequency:'
)

include_quality_bands = Checkbox(value=True, description='Include Quality Bands')
include_weather_data = Checkbox(value=False, description='Include Weather Data (slower)')

# Buttons
add_button = Button(description="Add Farm to List", button_style='primary')
enhanced_collection_button = Button(description="🚀 Enhanced Collection for ML", button_style='success')
quick_collection_button = Button(description="⚡ Quick Collection", button_style='info')
output_widget = Output()

def calculate_enhanced_indices(image):
    """Calculate comprehensive vegetation and soil indices"""
    
    # Basic vegetation indices
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    # SAVI (Soil Adjusted Vegetation Index) - better for arid regions
    savi = image.expression(
        '1.5 * (NIR - RED) / (NIR + RED + 0.5)', 
        {'NIR': image.select('B8'), 'RED': image.select('B4')}
    ).rename('SAVI')
    
    # EVI (Enhanced Vegetation Index) - reduces atmospheric interference
    evi = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }
    ).rename('EVI')
    
    # MSI (Moisture Stress Index)
    msi = image.normalizedDifference(['B11', 'B8']).rename('MSI')
    
    # Chlorophyll Red Edge
    chl_red_edge = image.normalizedDifference(['B7', 'B5']).rename('CHL_RED_EDGE')
    
    # BSI (Bare Soil Index)
    bsi = image.expression(
        '(RED + SWIR1) - (NIR + BLUE) / (RED + SWIR1) + (NIR + BLUE)',
        {
            'RED': image.select('B4'),
            'SWIR1': image.select('B11'),
            'NIR': image.select('B8'),
            'BLUE': image.select('B2')
        }
    ).rename('BSI')
    
    return image.addBands([ndvi, ndwi, savi, evi, msi, chl_red_edge, bsi])

def enhanced_data_collection(farm_aoi, farm_name, start_date, end_date, cloud_thresh, freq, include_quality):
    """
    Enhanced data collection with comprehensive indices and quality control
    """
    print(f"  🔍 Collecting enhanced satellite data for {farm_name}...")
    print(f"  📅 Date range: {start_date} to {end_date}")
    print(f"  ☁️ Cloud threshold: {cloud_thresh}%")
    
    try:
        # Build collection with filters
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(farm_aoi)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
                      .sort('system:time_start'))
        
        total_images = collection.size().getInfo()
        print(f"  📊 Found {total_images} images matching criteria")
        
        if total_images == 0:
            print(f"  ❌ No images found for {farm_name}")
            return pd.DataFrame()
        
        # Apply frequency filter
        if freq == 'monthly':
            collection = collection.filter(ee.Filter.dayOfMonth().eq(1))
        elif freq == 'weekly':
            collection = collection.filter(ee.Filter.dayOfWeek().eq(1))
        
        filtered_count = collection.size().getInfo()
        print(f"  🔄 After frequency filter: {filtered_count} images")
        
        if filtered_count == 0:
            print(f"  ⚠️ No images after frequency filtering for {farm_name}")
            return pd.DataFrame()
        
        # Calculate indices
        indices_collection = collection.map(calculate_enhanced_indices)
        
        def get_comprehensive_stats(image):
            """Extract comprehensive statistics for each image"""
            
            # Mean values
            mean_dict = image.reduceRegion(
                reducer=ee.Reducer.mean(), 
                geometry=farm_aoi, 
                scale=20, 
                maxPixels=1e8
            )
            
            # Standard deviation for quality assessment
            std_dict = image.reduceRegion(
                reducer=ee.Reducer.stdDev(), 
                geometry=farm_aoi, 
                scale=20, 
                maxPixels=1e8
            )
            
            # Date information
            date = ee.Date(image.get('system:time_start'))
            
            # Base properties
            properties = {
                'time': image.get('system:time_start'),
                'year': date.get('year'),
                'month': date.get('month'),
                'day': date.get('day'),
                'day_of_year': date.getRelative('day', 'year'),
                'week_of_year': date.getRelative('week', 'year'),
                'quarter': ee.Number(date.get('month')).subtract(1).divide(3).floor().add(1),
                
                # Vegetation indices
                'NDVI': mean_dict.get('NDVI'),
                'NDWI': mean_dict.get('NDWI'),
                'SAVI': mean_dict.get('SAVI'),
                'EVI': mean_dict.get('EVI'),
                'MSI': mean_dict.get('MSI'),
                'CHL_RED_EDGE': mean_dict.get('CHL_RED_EDGE'),
                'BSI': mean_dict.get('BSI'),
                
                # Quality metrics
                'cloud_percent': image.get('CLOUDY_PIXEL_PERCENTAGE'),
                'NDVI_std': std_dict.get('NDVI')
            }
            
            # Add quality bands if requested
            if include_quality:
                properties.update({
                    'NDWI_std': std_dict.get('NDWI'),
                    'SAVI_std': std_dict.get('SAVI'),
                    'data_quality_score': ee.Number(100).subtract(image.get('CLOUDY_PIXEL_PERCENTAGE'))
                })
            
            return ee.Feature(None, properties)
        
        # Process all images
        mean_fc = ee.FeatureCollection(indices_collection.map(get_comprehensive_stats))
        
        # Filter for valid data
        mean_fc = mean_fc.filter(ee.Filter.notNull(['NDVI', 'NDWI', 'SAVI']))
        
        valid_count = mean_fc.size().getInfo()
        print(f"  ✅ Valid observations: {valid_count}")
        
        if valid_count == 0:
            print(f"  ❌ No valid observations for {farm_name}")
            return pd.DataFrame()
        
        # Convert to pandas with robust error handling
        try:
            df = geemap.ee_to_df(mean_fc)
            print(f"  🔄 Successfully converted to DataFrame")
        except Exception as e:
            print(f"  ⚠️ geemap conversion failed: {e}")
            print(f"  🔄 Trying alternative conversion...")
            
            try:
                # Alternative conversion method
                data_list = mean_fc.getInfo()
                
                if 'features' in data_list and len(data_list['features']) > 0:
                    data_rows = []
                    for feature in data_list['features']:
                        if feature['properties']:
                            data_rows.append(feature['properties'])
                    
                    df = pd.DataFrame(data_rows)
                    print(f"  ✅ Alternative conversion successful")
                else:
                    print(f"  ❌ No features in collection")
                    return pd.DataFrame()
                    
            except Exception as e2:
                print(f"  ❌ Alternative conversion failed: {e2}")
                return pd.DataFrame()
        
        if not df.empty:
            # Enhanced data processing
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            df = df.sort_values('timestamp')
            
            # Add season mapping (Saudi climate)
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            # Calculate additional temporal features
            df['is_growing_season'] = df['month'].isin([3, 4, 5, 10, 11, 12]).astype(int)
            df['heat_stress_period'] = df['month'].isin([6, 7, 8, 9]).astype(int)
            
            # Data quality indicators
            df['high_quality'] = (df['cloud_percent'] < 10).astype(int)
            df['medium_quality'] = ((df['cloud_percent'] >= 10) & (df['cloud_percent'] < 20)).astype(int)
            
            print(f"  🎯 Final dataset: {len(df)} observations")
            print(f"  📅 Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
            print(f"  📊 Average NDVI: {df['NDVI'].mean():.3f}")
            print(f"  🌟 High quality observations: {df['high_quality'].sum()} ({df['high_quality'].mean()*100:.1f}%)")
            
        return df
        
    except Exception as e:
        print(f"  ❌ Error in enhanced collection: {e}")
        import traceback
        print(f"  🔍 Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def process_enhanced_collection(b):
    """Enhanced processing with comprehensive data collection"""
    with output_widget:
        clear_output(wait=True)
        
        if not farm_collection:
            print("❌ No farms added to the list yet.")
            return
        
        print(f"🚀 Starting Enhanced Data Collection")
        print(f"📊 Processing {len(farm_collection)} farms")
        print("=" * 50)
        
        # Get configuration
        start_date = date_start_input.value
        end_date = date_end_input.value
        cloud_thresh = cloud_threshold.value
        freq = collection_frequency.value
        include_quality = include_quality_bands.value
        
        print(f"⚙️ Configuration:")
        print(f"  📅 Date range: {start_date} to {end_date}")
        print(f"  ☁️ Cloud threshold: {cloud_thresh}%")
        print(f"  🔄 Collection frequency: {freq}")
        print(f"  📈 Quality bands: {include_quality}")
        print()
        
        all_farms_data = []
        successful_farms = []
        failed_farms = []
        
        for i, farm in enumerate(farm_collection, 1):
            farm_name = farm['name']
            
            print(f"🏗️ Processing Farm {i}/{len(farm_collection)}: {farm_name}")
            
            try:
                # Handle geometry
                if 'geojson' in farm:
                    geojson = farm['geojson']
                    if isinstance(geojson, dict):
                        if 'geometry' in geojson:
                            farm_aoi = ee.Geometry(geojson['geometry'])
                        else:
                            farm_aoi = ee.Geometry(geojson)
                    else:
                        farm_aoi = ee.Geometry(geojson)
                else:
                    print(f"  ❌ No geometry found for {farm_name}")
                    failed_farms.append(farm_name)
                    continue
                
                # Create farm folder
                farm_folder = os.path.join(gdrive_folder, farm_name)
                if not os.path.exists(farm_folder):
                    os.makedirs(farm_folder)
                
                # Collect data
                df = enhanced_data_collection(
                    farm_aoi, farm_name, start_date, end_date, 
                    cloud_thresh, freq, include_quality
                )
                
                if not df.empty:
                    # Add farm metadata
                    df['farm_id'] = farm_name
                    df['farm_name'] = farm_name
                    df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Save individual farm data
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_filename = os.path.join(farm_folder, f'{farm_name}_enhanced_{timestamp}.csv')
                    df.to_csv(csv_filename, index=False)
                    print(f"  💾 Saved: {csv_filename}")
                    
                    # Add to consolidated dataset
                    all_farms_data.append(df)
                    successful_farms.append(farm_name)
                    
                    # Generate and save summary statistics
                    summary_stats = {
                        'farm_name': farm_name,
                        'collection_date': datetime.now().isoformat(),
                        'total_observations': len(df),
                        'date_range_start': df['timestamp'].min().isoformat(),
                        'date_range_end': df['timestamp'].max().isoformat(),
                        'years_covered': df['year'].nunique(),
                        'avg_ndvi': float(df['NDVI'].mean()),
                        'avg_ndwi': float(df['NDWI'].mean()),
                        'avg_savi': float(df['SAVI'].mean()),
                        'ndvi_range': [float(df['NDVI'].min()), float(df['NDVI'].max())],
                        'data_quality': {
                            'avg_cloud_percent': float(df['cloud_percent'].mean()),
                            'high_quality_obs': int(df.get('high_quality', pd.Series([0])).sum()),
                            'quality_score': float((1 - df['cloud_percent'].mean()/100) * 100)
                        },
                        'seasonal_stats': {
                            'spring_avg_ndvi': float(df[df['season'] == 'Spring']['NDVI'].mean()),
                            'summer_avg_ndvi': float(df[df['season'] == 'Summer']['NDVI'].mean()),
                            'fall_avg_ndvi': float(df[df['season'] == 'Fall']['NDVI'].mean()),
                            'winter_avg_ndvi': float(df[df['season'] == 'Winter']['NDVI'].mean())
                        }
                    }
                    
                    # Save summary
                    summary_filename = os.path.join(farm_folder, f'{farm_name}_summary_{timestamp}.json')
                    with open(summary_filename, 'w') as f:
                        json.dump(summary_stats, f, indent=2, default=str)
                    
                    print(f"  📊 Summary saved: {summary_filename}")
                    
                else:
                    print(f"  ❌ No data collected for {farm_name}")
                    failed_farms.append(farm_name)
                
                # Add small delay to avoid rate limiting
                time.sleep(2)
                    
            except Exception as e:
                print(f"  ❌ Error processing {farm_name}: {e}")
                failed_farms.append(farm_name)
                continue
        
        print("\n" + "="*50)
        print("📋 COLLECTION SUMMARY")
        print("="*50)
        
        # Create consolidated dataset
        if all_farms_data:
            print(f"✅ Successfully processed: {len(successful_farms)} farms")
            print(f"❌ Failed: {len(failed_farms)} farms")
            
            if failed_farms:
                print(f"   Failed farms: {', '.join(failed_farms)}")
            
            consolidated_df = pd.concat(all_farms_data, ignore_index=True)
            consolidated_df = consolidated_df.sort_values(['farm_id', 'timestamp'])
            
            # Create timestamp for files
            file_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save consolidated dataset
            consolidated_filename = os.path.join(gdrive_folder, f'consolidated_enhanced_farm_data_{file_timestamp}.csv')
            consolidated_df.to_csv(consolidated_filename, index=False)
            
            # Create ML-ready dataset
            ml_ready_df = consolidated_df.copy()
            ml_ready_df['time'] = ml_ready_df['timestamp'].astype('int64') // 10**6
            
            # Organize columns for ML pipeline
            id_cols = ['time', 'timestamp', 'farm_id', 'farm_name']
            feature_cols = ['NDVI', 'NDWI', 'SAVI', 'EVI', 'MSI', 'CHL_RED_EDGE', 'BSI']
            temporal_cols = ['year', 'month', 'day', 'day_of_year', 'week_of_year', 'quarter', 'season']
            quality_cols = ['cloud_percent', 'NDVI_std', 'high_quality', 'data_quality_score']
            derived_cols = ['is_growing_season', 'heat_stress_period']
            
            # Include only existing columns
            available_cols = []
            for col_group in [id_cols, feature_cols, temporal_cols, quality_cols, derived_cols]:
                available_cols.extend([col for col in col_group if col in ml_ready_df.columns])
            
            ml_dataset = ml_ready_df[available_cols]
            ml_filename = os.path.join(gdrive_folder, f'ml_ready_farm_data_{file_timestamp}.csv')
            ml_dataset.to_csv(ml_filename, index=False)
            
            # Generate comprehensive summary report
            total_obs = len(consolidated_df)
            date_range = f"{consolidated_df['timestamp'].min().strftime('%Y-%m-%d')} to {consolidated_df['timestamp'].max().strftime('%Y-%m-%d')}"
            farms_count = consolidated_df['farm_id'].nunique()
            avg_ndvi = consolidated_df['NDVI'].mean()
            
            print(f"\n📊 CONSOLIDATED DATASET STATISTICS:")
            print(f"   Total observations: {total_obs:,}")
            print(f"   Farms included: {farms_count}")
            print(f"   Date range: {date_range}")
            print(f"   Average NDVI: {avg_ndvi:.3f}")
            print(f"   Features available: {len([col for col in feature_cols if col in consolidated_df.columns])}")
            
            # Quality statistics
            if 'cloud_percent' in consolidated_df.columns:
                avg_cloud = consolidated_df['cloud_percent'].mean()
                high_quality_pct = consolidated_df.get('high_quality', pd.Series([0])).mean() * 100
                print(f"   Average cloud coverage: {avg_cloud:.1f}%")
                print(f"   High quality observations: {high_quality_pct:.1f}%")
            
            print(f"\n💾 FILES CREATED:")
            print(f"   📄 Consolidated dataset: {os.path.basename(consolidated_filename)}")
            print(f"   🤖 ML-ready dataset: {os.path.basename(ml_filename)}")
            print(f"   📁 Individual farm data: [farm_name]_enhanced_[timestamp].csv")
            print(f"   📋 Farm summaries: [farm_name]_summary_[timestamp].json")
            
            # Create metadata file
            metadata = {
                'collection_info': {
                    'collection_date': datetime.now().isoformat(),
                    'script_version': 'enhanced_v2.0',
                    'total_farms_processed': len(farm_collection),
                    'successful_farms': len(successful_farms),
                    'failed_farms': len(failed_farms),
                    'configuration': {
                        'start_date': start_date,
                        'end_date': end_date,
                        'cloud_threshold': cloud_thresh,
                        'frequency': freq,
                        'include_quality_bands': include_quality
                    }
                },
                'dataset_info': {
                    'total_observations': total_obs,
                    'farms_count': farms_count,
                    'date_range': date_range,
                    'features_available': len([col for col in feature_cols if col in consolidated_df.columns]),
                    'files_created': {
                        'consolidated': os.path.basename(consolidated_filename),
                        'ml_ready': os.path.basename(ml_filename)
                    }
                },
                'quality_metrics': {
                    'average_ndvi': float(avg_ndvi),
                    'average_cloud_percent': float(consolidated_df.get('cloud_percent', pd.Series([0])).mean()),
                    'high_quality_percentage': float(consolidated_df.get('high_quality', pd.Series([0])).mean() * 100)
                }
            }
            
            metadata_filename = os.path.join(gdrive_folder, f'collection_metadata_{file_timestamp}.json')
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"   📋 Metadata: {os.path.basename(metadata_filename)}")
            
        else:
            print("❌ No data collected from any farms")
            print("Check farm geometries and date ranges")
        
        print(f"\n🎉 Enhanced collection completed!")
        print(f"📁 Check Google Drive folder: {gdrive_folder}")

def quick_collection_process(b):
    """Quick collection with basic indices for fast testing"""
    with output_widget:
        clear_output(wait=True)
        
        if not farm_collection:
            print("❌ No farms added to the list yet.")
            return
        
        print(f"⚡ Starting Quick Data Collection")
        print(f"📊 Processing {len(farm_collection)} farms")
        print("=" * 40)
        
        for i, farm in enumerate(farm_collection, 1):
            farm_name = farm['name']
            print(f"🏗️ Quick processing {i}/{len(farm_collection)}: {farm_name}")
            
            try:
                farm_aoi = ee.Geometry(farm['geojson'])
                farm_folder = os.path.join(gdrive_folder, farm_name)
                if not os.path.exists(farm_folder):
                    os.makedirs(farm_folder)
                
                # Quick collection - last 2 years only
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                              .filterBounds(farm_aoi)
                              .filterDate('2023-01-01', '2024-12-31')
                              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
                
                def calculate_basic_indices(image):
                    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
                    savi = image.expression('1.5 * (NIR - RED) / (NIR + RED + 0.5)', 
                                            {'NIR': image.select('B8'), 'RED': image.select('B4')}).rename('SAVI')
                    return image.addBands([ndvi, ndwi, savi])

                def get_basic_stats(image):
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

                indices_collection = collection.map(calculate_basic_indices)
                mean_fc = ee.FeatureCollection(indices_collection.map(get_basic_stats))
                mean_fc = mean_fc.filter(ee.Filter.notNull(['NDVI', 'NDWI', 'SAVI']))
                
                df = geemap.ee_to_df(mean_fc)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                    df['farm_id'] = farm_name
                    df = df.sort_values('timestamp')
                    
                    quick_filename = os.path.join(farm_folder, f'{farm_name}_quick_collection.csv')
                    df.to_csv(quick_filename, index=False)
                    print(f"  ✅ Saved {len(df)} observations")
                else:
                    print(f"  ❌ No data for {farm_name}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("⚡ Quick collection completed!")

# Function to add farm to list
def add_farm_to_list(b):
    with output_widget:
        if not m.user_roi:
            print("❌ Please draw a boundary on the map first.")
            return
        if not farm_name_input.value:
            print("❌ Please enter a farm name.")
            return
        
        farm_name = farm_name_input.value
        farm_geojson = m.user_roi
        farm_collection.append({'name': farm_name, 'geojson': farm_geojson})
        
        print(f"✅ Farm '{farm_name}' added. Total farms: {len(farm_collection)}")
        farm_name_input.value = ''
        m.remove_drawn_features()

# Link functions to buttons
add_button.on_click(add_farm_to_list)
enhanced_collection_button.on_click(process_enhanced_collection)
quick_collection_button.on_click(quick_collection_process)

# Create UI layout
print("\n🎛️ CONFIGURATION PANEL")
config_panel = VBox([
    HBox([date_start_input, date_end_input]),
    HBox([cloud_threshold, collection_frequency]),
    HBox([include_quality_bands, include_weather_data])
])

print("\n🗺️ MAP AND CONTROLS")
control_panel = VBox([
    m,
    HBox([farm_name_input, add_button]),
    HBox([quick_collection_button, enhanced_collection_button])
])

# Display the complete interface
ui = VBox([
    config_panel,
    control_panel,
    output_widget
])

print("\n" + "="*60)
print("🌴 PALM FARM DATA COLLECTION SYSTEM READY")
print("="*60)
print("1. Configure collection parameters above")
print("2. Draw farm boundaries on the map")
print("3. Enter farm names and add to list")
print("4. Choose Quick Collection (test) or Enhanced Collection (full)")
print("5. Data will be saved to your Google Drive")
print("\n📁 Output folder: " + gdrive_folder)

display(ui)
