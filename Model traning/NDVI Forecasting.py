import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
def load_data(file_path="/content/drive/MyDrive/palm/data/consolidated_historical_2015_2025.csv"):
 
# def load_data(file_path="consolidated_historical_2015_2025.csv"):
    """Loads and preprocesses the farm data."""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['farm_name', 'timestamp']).reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory as the script.")
        return pd.DataFrame()


def feature_engineering_for_forecasting(df):
    """Creates time-series features for the forecasting model."""
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    
    # Create lag features (NDVI from previous periods)
    df['ndvi_lag_1'] = df.groupby('farm_name')['NDVI'].shift(1)
    df['ndvi_lag_7'] = df.groupby('farm_name')['NDVI'].shift(7)
    
    # Create rolling window features
    df['ndvi_roll_mean_7'] = df.groupby('farm_name')['NDVI'].shift(1).rolling(window=7).mean()
    df['ndvi_roll_std_7'] = df.groupby('farm_name')['NDVI'].shift(1).rolling(window=7).std()

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['season', 'farm_name'], drop_first=True)
    
    df = df.dropna().reset_index(drop=True)
    return df

def run_anomaly_detection(df):
    """
    Trains an Isolation Forest model to detect anomalies in NDVI data.
    """
    print("\n--- Running Anomaly Detection using Isolation Forest ---")
    
    # Select features for anomaly detection
    features = ['NDVI', 'NDWI', 'SAVI', 'cloud_percent']
    data_for_anomaly = df[features]
    
    # Initialize and train the model
    # contamination='auto' is a good starting point
    iso_forest = IsolationForest(contamination='auto', random_state=42)
    iso_forest.fit(data_for_anomaly)
    
    # Predict anomalies (-1 for anomalies, 1 for inliers)
    df['anomaly_score'] = iso_forest.decision_function(data_for_anomaly)
    df['is_anomaly'] = iso_forest.predict(data_for_anomaly)
    
    anomalies = df[df['is_anomaly'] == -1]
    print(f"Found {len(anomalies)} potential anomalies out of {len(df)} records.")
    
    # Display the top 10 anomalies
    print("\nTop 10 most anomalous data points:")
    print(anomalies.sort_values('anomaly_score').head(10)[['timestamp', 'farm_name', 'NDVI', 'anomaly_score']])

    # Visualize the results for a sample farm
    sample_farm = df['farm_name'].unique()[0]
    plt.figure(figsize=(15, 6))
    farm_df = df[df['farm_name'] == sample_farm]
    
    inliers = farm_df[farm_df['is_anomaly'] == 1]
    outliers = farm_df[farm_df['is_anomaly'] == -1]
    
    plt.scatter(inliers['timestamp'], inliers['NDVI'], c='lightgreen', label='Normal')
    plt.scatter(outliers['timestamp'], outliers['NDVI'], c='red', marker='x', label='Anomaly')
    
    plt.title(f'NDVI Anomaly Detection for {sample_farm}')
    plt.xlabel('Date')
    plt.ylabel('NDVI')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_ndvi_forecasting(df):
    """
    Trains a LightGBM model to forecast future NDVI values.
    """
    print("\n--- Running NDVI Forecasting using LightGBM ---")
    
    # Engineer features
    df_featured = feature_engineering_for_forecasting(df)
    
    # Define features (X) and target (y)
    target = 'NDVI'
    # FIXED: Exclude the non-numeric 'satellite' column from the features
    features = [col for col in df_featured.columns if col not in ['timestamp', 'farm_id', 'NDVI', 'satellite']]

    X = df_featured[features]
    y = df_featured[target]
    
    # Split data into training and testing sets (time-based split)
    split_date = df_featured['timestamp'].max() - pd.DateOffset(months=6)
    train_idx = df_featured[df_featured['timestamp'] < split_date].index
    test_idx = df_featured[df_featured['timestamp'] >= split_date].index

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Initialize and train the LightGBM model
    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgb_model.fit(X_train, y_train)
    
    # Make predictions
    predictions = lgb_model.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"\nModel Evaluation (Root Mean Squared Error): {rmse:.4f}")

    # Visualize predictions vs actuals for a sample farm
    test_df = df_featured.loc[test_idx].copy()
    test_df['prediction'] = predictions
    
    # Find a farm name that exists in the test set for plotting
    plot_farm_name = None
    for farm in df['farm_name'].unique():
        if f'farm_name_{farm}' in test_df.columns:
            if test_df[f'farm_name_{farm}'].sum() > 0:
                plot_farm_name = farm
                break
    
    if plot_farm_name:
        farm_test_df = test_df[test_df[f'farm_name_{plot_farm_name}'] == 1]
        plt.figure(figsize=(15, 6))
        plt.plot(farm_test_df['timestamp'], farm_test_df['NDVI'], label='Actual NDVI', marker='.')
        plt.plot(farm_test_df['timestamp'], farm_test_df['prediction'], label='Predicted NDVI', linestyle='--')
        plt.title(f'NDVI Forecast vs. Actual for {plot_farm_name}')
        plt.xlabel('Date')
        plt.ylabel('NDVI')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Could not find a suitable farm in the test set to plot.")


if __name__ == '__main__':
    # Load the data
    farm_data = load_data()
    
    if not farm_data.empty:
        # Run the anomaly detection model
        run_anomaly_detection(farm_data.copy())
        
        # Run the NDVI forecasting model
        run_ndvi_forecasting(farm_data.copy())
