import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = "consolidated_historical_2015_2025.csv"
MODEL_FILE = "lgb_model.joblib"
FEATURES_FILE = "forecasting_features.joblib"

def robust_feature_engineering(df):
    """Creates robust, non-leaky time-series features for forecasting."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time-based features
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    
    # Cyclical features for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # One-hot encode farm names
    df_featured = pd.get_dummies(df, columns=['farm_name'], drop_first=True)
    return df_featured

def train_forecasting_model():
    """Trains and saves a robust forecasting model using a proper time-based split."""
    print("--- 1. Loading and Preparing Data for Training ---")
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        print("Data file is empty. Cannot train.")
        return

    print("\n--- 2. Engineering Robust Features ---")
    df_featured = robust_feature_engineering(df)

    # --- THE CRITICAL FIX: TIME-BASED SPLIT BEFORE TRAINING ---
    print("\n--- 3. Splitting Data: Training on data BEFORE the last year ---")
    split_date = df_featured['timestamp'].max() - pd.DateOffset(years=1)
    
    train_df = df_featured[df_featured['timestamp'] < split_date].copy()
    
    print(f"Training data will be from {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}")

    target = 'NDVI'
    features = [col for col in train_df.columns if col not in ['timestamp', 'farm_id', 'NDVI', 'season', 'satellite']]
    
    # Ensure all features exist, just in case
    features = [f for f in features if f in train_df.columns]
    
    X_train = train_df[features]
    y_train = train_df[target]

    print(f"Training robust model on {len(X_train)} records...")

    # Train the model ONLY on the training data
    lgb_model = lgb.LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.05)
    lgb_model.fit(X_train, y_train)
    
    # Save the correctly trained model and the feature list
    joblib.dump(lgb_model, MODEL_FILE)
    joblib.dump(features, FEATURES_FILE)
    print("\n--- 4. New, Correctly Trained Model Saved Successfully ---")
    print(f"'{MODEL_FILE}' and '{FEATURES_FILE}' have been updated.")

if __name__ == "__main__":
    train_forecasting_model()
