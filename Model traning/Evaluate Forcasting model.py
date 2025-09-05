import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import warnings

warnings.filterwarnings('ignore')

# --- 1. SETUP ---
print("--- 1. Setting Up Environment ---")
drive.mount('/content/drive', force_remount=True)

# --- Configuration ---
DATA_FILE = "/content/drive/MyDrive/palm/data/consolidated_historical_2015_2025.csv"
MODEL_FILE = "lgb_model.joblib"
FEATURES_FILE = "forecasting_features.joblib"

def robust_feature_engineering(df):
    """Creates robust, non-leaky time-series features for forecasting."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    df_featured = pd.get_dummies(df, columns=['farm_name'], drop_first=True)
    return df_featured

def plot_actual_vs_predicted(test_df, farm_name):
    """Visualizes the actual vs. predicted values for a specific farm."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    farm_test_data = test_df[test_df['farm_name'] == farm_name]
    ax.plot(farm_test_data['timestamp'], farm_test_data['NDVI'], label='Actual NDVI', color='green', marker='.', markersize=4)
    ax.plot(farm_test_data['timestamp'], farm_test_data['prediction'], label='Predicted NDVI', color='red', linestyle='--')
    ax.set_title(f'Model Performance for {farm_name}', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('NDVI')
    ax.legend()
    plt.show()

def evaluate_robust_model():
    """Main function to run the complete model evaluation."""
    print("\n--- 2. Loading Models, Features, and Data ---")
    model = joblib.load(MODEL_FILE)
    features_list = joblib.load(FEATURES_FILE)
    df = pd.read_csv(DATA_FILE)
    if df.empty: return

    print("\n--- 3. Engineering Robust Features on Full Dataset ---")
    df_featured = robust_feature_engineering(df)
    
    # Align columns to match the model's training features exactly
    X = pd.DataFrame(columns=features_list)
    X = pd.concat([X, df_featured], ignore_index=False).fillna(0)
    X = X[features_list]
    y = df_featured['NDVI']

    print("\n--- 4. Splitting Data: Testing on the last year of data ---")
    split_date = df_featured['timestamp'].max() - pd.DateOffset(years=1)
    test_idx = df_featured[df_featured['timestamp'] >= split_date].index
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    if X_test.empty:
        print("❌ Error: Test set is empty.")
        return
    print(f"Test data size: {len(X_test)} records from {split_date.date()} onwards.")

    predictions = model.predict(X_test)
    
    print("\n--- 5. Overall Model Performance Metrics ---")
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE):      {mae:.4f}")
    print(f"  - R-squared (R²):                 {r2:.4f}")

    print("\n--- 6. Analyzing Per-Farm Results ---")
    test_results = df_featured.loc[test_idx].copy()
    test_results['prediction'] = predictions
    test_results['error'] = test_results['NDVI'] - test_results['prediction']
    
    farm_metrics = []
    # Use the original farm_name column from the source dataframe
    test_results['farm_name'] = df.loc[test_idx, 'farm_name']
    
    for farm_name in test_results['farm_name'].unique():
        farm_data = test_results[test_results['farm_name'] == farm_name]
        if not farm_data.empty:
            farm_r2 = r2_score(farm_data['NDVI'], farm_data['prediction'])
            farm_metrics.append({'farm_name': farm_name, 'r2': farm_r2})

    summary_df = pd.DataFrame(farm_metrics).sort_values(by='r2', ascending=False)
    print("\nPerformance (R²) by Farm on Test Set:")
    print(summary_df.to_string(index=False))
    
    print("\n--- 7. Visualizing Results ---")
    best_farm_name = summary_df.iloc[0]['farm_name']
    worst_farm_name = summary_df.iloc[-1]['farm_name']
    
    plot_actual_vs_predicted(test_results, best_farm_name)
    plot_actual_vs_predicted(test_results, worst_farm_name)

    print("\n--- Evaluation Complete ---")

if __name__ == '__main__':
    evaluate_robust_model()
