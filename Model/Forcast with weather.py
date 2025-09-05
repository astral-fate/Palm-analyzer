import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import optuna
from sklearn.metrics import mean_squared_error
import requests
import warnings

warnings.filterwarnings('ignore')

# --- 1. Configuration ---
DATA_FILE = "consolidated_historical_2015_2025.csv"
MODEL_FILE = "lgb_model.joblib"
FEATURES_FILE = "forecasting_features.joblib"

# Define the coordinates for the weather data
LOCATION_COORDS = {"latitude": 24.47, "longitude": 39.61} # Madinah, Saudi Arabia

# --- 2. Data Fetching and Preparation ---

def fetch_weather_data(lat, lon, start_date, end_date):
    """Fetches historical daily weather data from the Open-Meteo API."""
    print(f"Fetching historical weather data from {start_date} to {end_date}...")
    URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,shortwave_radiation_sum"
    }
    try:
        response = requests.get(URL, params=params)
        response.raise_for_status()
        data = response.json()['daily']
        weather_df = pd.DataFrame(data)
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        print("✅ Weather data fetched successfully.")
        return weather_df
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch weather data: {e}")
        return None

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

# --- 3. Optuna Optimization ---

def objective(trial, X_train, y_train, X_val, y_val):
    """The function for Optuna to optimize."""
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

# --- 4. Main Training Workflow ---

def train_optimized_model():
    """Main function to run the complete training and optimization workflow."""
    print("--- 1. Loading and Preparing Data ---")
    df = pd.read_csv(DATA_FILE)
    if df.empty: return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    weather_df = fetch_weather_data(
        LOCATION_COORDS['latitude'],
        LOCATION_COORDS['longitude'],
        df['timestamp'].min().strftime('%Y-%m-%d'),
        df['timestamp'].max().strftime('%Y-%m-%d')
    )
    if weather_df is not None:
        df = pd.merge(df, weather_df, left_on=df['timestamp'].dt.date, right_on=weather_df['time'].dt.date, how='left')
        cols_to_drop = ['key_0', 'time']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
        df.ffill(inplace=True)

    print("\n--- 2. Engineering Robust Features ---")
    df_featured = robust_feature_engineering(df)

    print("\n--- 3. Splitting Data for Optimization ---")
    split_date = df_featured['timestamp'].max() - pd.DateOffset(years=1)
    train_df = df_featured[df_featured['timestamp'] < split_date].copy()
    val_df = df_featured[df_featured['timestamp'] >= split_date].copy()

    target = 'NDVI'
    
    # --- FINAL FIX: Select features by numeric data type ---
    # This guarantees no non-numeric columns like 'timestamp' are included.
    features = train_df.select_dtypes(include=np.number).columns.tolist()
    
    # Now, explicitly remove the target and any other non-feature columns.
    features_to_remove = [target, 'year', 'month', 'day', 'cloud_percent', 'NDWI', 'SAVI']
    features = [f for f in features if f not in features_to_remove]
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    print(f"Training on {len(X_train)} records, validating on {len(X_val)} records.")
    print(f"Number of features used: {len(features)}")

    print("\n--- 4. Starting Hyperparameter Optimization with Optuna ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)

    print(f"Optimization finished. Best RMSE: {study.best_value:.4f}")
    print("Best parameters found:")
    print(study.best_params)

    print("\n--- 5. Training Final Model with Best Parameters ---")
    final_model = lgb.LGBMRegressor(random_state=42, **study.best_params)
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(features, FEATURES_FILE)
    print(f"\n✅ New optimized model saved as '{MODEL_FILE}'")
    print(f"✅ Feature list saved as '{FEATURES_FILE}'")

if __name__ == "__main__":
    train_optimized_model()
