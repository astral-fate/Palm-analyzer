import pandas as pd
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = "consolidated_historical_2015_2025.csv"

def create_forecasting_model(df):
    """Trains and saves the LightGBM forecasting model."""
    print("Training and saving forecasting model...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature Engineering
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    
    df = df.sort_values(by=['farm_name', 'timestamp'])
    df['ndvi_lag_1'] = df.groupby('farm_name')['NDVI'].shift(1)
    df['ndvi_lag_7'] = df.groupby('farm_name')['NDVI'].shift(7)
    df['ndvi_roll_mean_7'] = df.groupby('farm_name')['NDVI'].shift(1).rolling(window=7, min_periods=1).mean()
    df['ndvi_roll_std_7'] = df.groupby('farm_name')['NDVI'].shift(1).rolling(window=7, min_periods=1).std()

    # One-hot encode categorical features
    df_featured = pd.get_dummies(df, columns=['season', 'farm_name'], drop_first=True)
    df_featured = df_featured.dropna()

    target = 'NDVI'
    # Exclude non-feature columns
    features = [col for col in df_featured.columns if col not in ['timestamp', 'farm_id', 'NDVI', 'satellite']]
    
    X = df_featured[features]
    y = df_featured[target]

    # Train the model
    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgb_model.fit(X, y)
    
    # Save the model and feature list
    joblib.dump(lgb_model, 'lgb_model.joblib')
    joblib.dump(features, 'forecasting_features.joblib')
    print("Forecasting model and feature list saved successfully.")

if __name__ == "__main__":
    main_df = pd.read_csv(DATA_FILE)
    create_forecasting_model(main_df)
    print("\n'lgb_model.joblib' and 'forecasting_features.joblib' are now ready to be uploaded to Hugging Face.")
