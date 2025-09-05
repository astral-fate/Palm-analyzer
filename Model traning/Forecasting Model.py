import pandas as pd
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = "consolidated_historical_2015_2025.csv"
MODEL_FILE = "lgb_model.joblib"
FEATURES_FILE = "forecasting_features.joblib"

def feature_engineering(df):
    """Creates time-series features for the forecasting model."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
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

    df_featured = pd.get_dummies(df, columns=['season', 'farm_name'], drop_first=True)
    df_featured = df_featured.dropna().reset_index(drop=True)
    return df_featured

def train_forecasting_model():
    """Trains and saves the LightGBM model using a proper time-based split."""
    print("--- 1. Loading and Preparing Data for Training ---")
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        print("Data file is empty. Cannot train.")
        return

    print("\n--- 2. Engineering Features ---")
    df_featured = feature_engineering(df)

    # --- THE CRITICAL FIX: TIME-BASED SPLIT ---
    print("\n--- 3. Splitting Data: Training on data before the last year ---")
    split_date = df_featured['timestamp'].max() - pd.DateOffset(years=1)
    
    train_df = df_featured[df_featured['timestamp'] < split_date]
    
    print(f"Training data will be from {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")

    target = 'NDVI'
    features = [col for col in train_df.columns if col not in ['timestamp', 'farm_id', 'NDVI', 'satellite']]
    
    X_train = train_df[features]
    y_train = train_df[target]

    print(f"Training model on {len(X_train)} records...")

    # Train the model ONLY on the training data
    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgb_model.fit(X_train, y_train)
    
    # Save the correctly trained model and the feature list
    joblib.dump(lgb_model, MODEL_FILE)
    joblib.dump(features, FEATURES_FILE)
    print("\n--- 4. New Model Saved Successfully ---")
    print(f"'{MODEL_FILE}' and '{FEATURES_FILE}' have been updated.")

if __name__ == "__main__":
    train_forecasting_model()
