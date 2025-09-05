import pandas as pd
import os
from pathlib import Path

def consolidate_farm_data(root_folder, output_filename="consolidated_farm_data.csv"):
    """
    Consolidate all farm CSV files from nested subfolders into one master CSV file.
    
    Parameters:
    root_folder (str): Path to the main folder containing farm subfolders
    output_filename (str): Name of the output consolidated CSV file
    
    Expected structure:
    root_folder/
    ├── farm1/
    │   └── farm1_indices_timeseries.csv
    ├── farm2/
    │   └── farm2_indices_timeseries.csv
    └── ...
    """
    
    all_data = []
    processed_farms = []
    errors = []
    
    # Convert to Path object for easier handling
    root_path = Path(root_folder)
    
    if not root_path.exists():
        print(f"Error: Folder '{root_folder}' not found!")
        return
    
    # Get all subdirectories (farm folders)
    farm_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if not farm_folders:
        print(f"No subfolders found in '{root_folder}'")
        return
    
    print(f"Found {len(farm_folders)} farm folders. Processing...")
    
    for farm_folder in farm_folders:
        farm_name = farm_folder.name
        print(f"Processing farm: {farm_name}")
        
        # Look for CSV files in the farm folder
        csv_files = list(farm_folder.glob("*.csv"))
        
        if not csv_files:
            errors.append(f"No CSV files found in {farm_name}")
            continue
        
        # Try to find the timeseries file (prefer the one with farm name)
        timeseries_file = None
        
        # First, try to find file with farm name pattern
        for csv_file in csv_files:
            if f"{farm_name}_indices_timeseries.csv" == csv_file.name:
                timeseries_file = csv_file
                break
            elif "timeseries" in csv_file.name.lower():
                timeseries_file = csv_file
                break
        
        # If no timeseries file found, use the first CSV
        if not timeseries_file:
            timeseries_file = csv_files[0]
        
        try:
            # Read the CSV file
            df = pd.read_csv(timeseries_file)
            
            # Add farm identifier column
            df['farm_id'] = farm_name
            df['farm_name'] = farm_name  # Keep both for flexibility
            
            # Ensure consistent column order (farm info first)
            cols = ['farm_id', 'farm_name'] + [col for col in df.columns if col not in ['farm_id', 'farm_name']]
            df = df[cols]
            
            # Add to master list
            all_data.append(df)
            processed_farms.append(farm_name)
            print(f"  ✓ Successfully processed {farm_name} ({len(df)} records)")
            
        except Exception as e:
            error_msg = f"Error processing {farm_name}: {str(e)}"
            errors.append(error_msg)
            print(f"  ✗ {error_msg}")
    
    # Combine all data
    if all_data:
        print(f"\nCombining data from {len(all_data)} farms...")
        consolidated_df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        output_path = root_path / output_filename
        consolidated_df.to_csv(output_path, index=False)
        
        print(f"\n🎉 SUCCESS!")
        print(f"Consolidated data saved to: {output_path}")
        print(f"Total records: {len(consolidated_df):,}")
        print(f"Total farms: {consolidated_df['farm_id'].nunique()}")
        print(f"Columns: {list(consolidated_df.columns)}")
        
        # Show summary statistics
        print(f"\n📊 SUMMARY:")
        print(f"Farms processed: {', '.join(processed_farms)}")
        
        if 'NDVI' in consolidated_df.columns:
            print(f"NDVI range: {consolidated_df['NDVI'].min():.3f} to {consolidated_df['NDVI'].max():.3f}")
            print(f"Average NDVI: {consolidated_df['NDVI'].mean():.3f}")
        
        # Show data preview
        print(f"\n📋 DATA PREVIEW:")
        print(consolidated_df.head())
        
        return output_path
    
    else:
        print("\n❌ No data could be processed!")
    
    # Show errors if any
    if errors:
        print(f"\n⚠️  ERRORS ENCOUNTERED:")
        for error in errors:
            print(f"  - {error}")

def create_summary_report(consolidated_csv_path):
    """
    Create a summary report of the consolidated data
    """
    df = pd.read_csv(consolidated_csv_path)
    
    print(f"\n📈 DETAILED ANALYSIS OF CONSOLIDATED DATA:")
    print(f"=" * 50)
    
    # Farm-wise statistics
    if 'NDVI' in df.columns:
        farm_stats = df.groupby('farm_id')['NDVI'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
        print(f"\nFarm-wise NDVI Statistics:")
        print(farm_stats)
    
    # Time range analysis
    if 'time' in df.columns:
        try:
            df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
            if df['time'].notna().any():
                print(f"\nTime Range: {df['time'].min()} to {df['time'].max()}")
                print(f"Total duration: {(df['time'].max() - df['time'].min()).days} days")
        except:
            print("\nCould not parse time column for analysis")
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    if missing_data.any():
        print(f"\nMissing Data:")
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                print(f"  {col}: {missing_count} ({missing_count/len(df)*100:.1f}%)")

# Example usage and main execution
if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    ROOT_FOLDER = "/content/drive/MyDrive/palm/data/"  # Change this to your actual path
    OUTPUT_FILENAME = "consolidated_palm_farm_data.csv"
    
    print("🌴 PALM FARM DATA CONSOLIDATOR")
    print("=" * 40)
    
    # Run the consolidation
    output_path = consolidate_farm_data(ROOT_FOLDER, OUTPUT_FILENAME)
    
    if output_path:
        # Create detailed report
        create_summary_report(output_path)
        
        print(f"\n✅ CONSOLIDATION COMPLETE!")
        print(f"You can now upload the single file: {output_path}")
        print(f"This file contains all your farm data with proper farm identification.")

# Alternative function for Google Colab users
def colab_consolidate():
    """
    Easy function for Google Colab users
    """
    from google.colab import drive
    
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Set your path here
    root_folder = input("Enter the path to your farm data folder (e.g., /content/drive/MyDrive/palm/data/): ")
    
    if not root_folder:
        root_folder = "/content/drive/MyDrive/palm/data/"
    
    output_filename = input("Enter output filename (or press Enter for default): ")
    if not output_filename:
        output_filename = "consolidated_palm_farm_data.csv"
    
    # Run consolidation
    output_path = consolidate_farm_data(root_folder, output_filename)
    
    if output_path:
        create_summary_report(output_path)
        
        # Offer to download
        print(f"\n💾 Download the consolidated file:")
        from google.colab import files
        try:
            files.download(str(output_path))
        except:
            print(f"File saved at: {output_path}")
            print("You can download it manually from the files panel")

# Uncomment the line below if running in Google Colab
# colab_consolidate()
