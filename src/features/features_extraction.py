import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

try:
    from src.feature_store.hopsworks_integration import save_features_to_hopsworks
    HOPSWORKS_ENABLED = True
except ImportError as e:
    HOPSWORKS_ENABLED = False
    print("Note: Hopsworks integration not available. Features will only be saved locally.")

# ==================== CONFIG ====================
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PAKISTAN_TZ = pytz.timezone('Asia/Karachi')

# ==================== AQI CALCULATION ====================
AQI_BREAKPOINTS = {
    "pm2_5": [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
               (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
              (255, 354, 151, 200), (355, 424, 201, 300)],
    "ozone": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
               (86, 105, 151, 200)],
    "nitrogen_dioxide": [(0, 53, 0, 50), (54, 100, 51, 100),
                          (101, 360, 101, 150), (361, 649, 151, 200)],
    "sulphur_dioxide": [(0, 35, 0, 50), (36, 75, 51, 100),
                         (76, 185, 101, 150), (186, 304, 151, 200)],
    "carbon_monoxide": [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100),
                         (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)]
}

AQI_CATEGORIES = {
    (0, 50): "Good",
    (51, 100): "Moderate",
    (101, 150): "Unhealthy for Sensitive Groups",
    (151, 200): "Unhealthy",
    (201, 300): "Very Unhealthy"
}


def compute_aqi_for_pollutant(conc, pollutant):
    """Compute AQI for a given pollutant concentration."""
    for (Clow, Chigh, Ilow, Ihigh) in AQI_BREAKPOINTS.get(pollutant, []):
        if Clow <= conc <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (conc - Clow) + Ilow
    return np.nan


def compute_overall_aqi(row):
    """Compute overall AQI as the max of individual pollutant AQIs."""
    pollutants = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]
    aqi_values = [compute_aqi_for_pollutant(row[p], p) for p in pollutants if not pd.isna(row[p])]
    return max(aqi_values) if aqi_values else np.nan


def categorize_aqi(aqi_value):
    """Assign AQI category."""
    for (low, high), category in AQI_CATEGORIES.items():
        if low <= aqi_value <= high:
            return category
    return "Hazardous"


def create_time_features(df):
    """Create time-based features from timestamp (Pakistan time aware)."""
    df = df.copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # If timestamp has timezone, ensure it's Pakistan time
    if df['timestamp'].dt.tz is not None:
        # Convert to Pakistan timezone if not already
        df['timestamp'] = df['timestamp'].dt.tz_convert(PAKISTAN_TZ)
    else:
        # If no timezone, assume UTC and convert to Pakistan time
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(PAKISTAN_TZ)
    
    print(f"   Using timezone: {df['timestamp'].dt.tz} (Pakistan Time)")
    
    # Time-based features (now in Pakistan time!)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def create_derived_features(df):
    """Create derived features from existing columns."""
    df = df.copy()
    
    # Sort by timestamp for rolling calculations
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Rolling statistics for pollutants
    for col in ['pm2_5', 'pm10', 'ozone']:
        df[f'{col}_rolling_mean_24h'] = df[col].shift(1).rolling(window=24, min_periods=1).mean()
        df[f'{col}_rolling_std_24h'] = df[col].shift(1).rolling(window=24, min_periods=1).std()
        df[f'{col}_rolling_max_24h'] = df[col].shift(1).rolling(window=24, min_periods=1).max()
    
    # Weather-derived features
    df['temperature_change'] = df['temperature_2m'].diff(1)
    df['wind_speed_change'] = df['wind_speed_10m'].diff(1)
    
    # Interaction features
    df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    df['temperature_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']
    df['wind_pressure_interaction'] = df['wind_speed_10m'] * df['surface_pressure']
    
    # Lag features
    for col in ['pm2_5', 'pm10', 'temperature_2m']:
        df[f'{col}_lag_1h'] = df[col].shift(1)
        df[f'{col}_lag_24h'] = df[col].shift(24)
    
    # Fill NaN values created by lag and diff operations
    df = df.bfill().ffill()
    
    return df


# ==================== MAIN ====================
def process_file(file_path, augment_std: float = 0.0, save_prefix: str = "processed_aqi"):
    """Process a single raw CSV file and save processed output."""
    print(f"Processing file: {file_path.name}")
    df = pd.read_csv(file_path)

    # Drop missing pollution values
    df.dropna(subset=["pm2_5", "pm10"], inplace=True)

    # Compute AQI and Category
    print("Computing AQI...")
    df["AQI"] = df.apply(compute_overall_aqi, axis=1)
    df["AQI_Category"] = df["AQI"].apply(categorize_aqi)

    # Drop rows without AQI
    df.dropna(subset=["AQI"], inplace=True)

    # Time and derived features
    print("Creating time-based features (Pakistan Time)...")
    df = create_time_features(df)
    
    print("Creating derived features...")
    df = create_derived_features(df)

    # Select final features for ML
    base_feature_cols = [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "temperature_2m",
        "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
        "wind_direction_10m", "precipitation"
    ]

    time_feature_cols = [
        "hour", "day_of_week", "day_of_month", "month", "quarter", "is_weekend",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos"
    ]

    derived_feature_cols = [
        "pm2_5_rolling_mean_24h", "pm2_5_rolling_std_24h", "pm2_5_rolling_max_24h",
        "pm10_rolling_mean_24h", "pm10_rolling_std_24h", "pm10_rolling_max_24h",
        "ozone_rolling_mean_24h", "ozone_rolling_std_24h", "ozone_rolling_max_24h",
        "temperature_change", "wind_speed_change",
        "pm2_5_pm10_ratio", "temperature_humidity_interaction", "wind_pressure_interaction",
        "pm2_5_lag_1h", "pm2_5_lag_24h",
        "pm10_lag_1h", "pm10_lag_24h", 
        "temperature_2m_lag_1h", "temperature_2m_lag_24h"
    ]

    all_feature_cols = base_feature_cols + time_feature_cols + derived_feature_cols
    available_feature_cols = [col for col in all_feature_cols if col in df.columns]

    # Optional augmentation
    if augment_std and augment_std > 0:
        print(f"Applying augmentation (std={augment_std})...")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        noise = pd.DataFrame(np.random.normal(0, augment_std, size=(len(df), len(numeric_cols))),
                             columns=numeric_cols, index=df.index)
        df[numeric_cols] = df[numeric_cols] + noise

    # Save processed file
    date_tag = file_path.stem.replace("openmeteo_combined_", "").replace("realtime_", "")
    processed_path = PROCESSED_DIR / f"{save_prefix}_{date_tag}.csv"
    df.to_csv(processed_path, index=False)

    # Save feature list
    feature_list_path = PROCESSED_DIR / "feature_list.txt"
    with open(feature_list_path, 'w') as f:
        f.write("Feature Columns (extracted in Pakistan Time):\n")
        for col in available_feature_cols:
            f.write(f"  - {col}\n")

    print(f"✓ Processed data saved to: {processed_path}")
    print(f"✓ Feature list saved to: {feature_list_path}")
    print(f"Rows: {len(df)}, Total Columns: {len(df.columns)}")
    print(f"Feature Columns: {len(available_feature_cols)}")
    print(f"AQI Range: {df['AQI'].min():.1f} - {df['AQI'].max():.1f}")
    print(f"Categories: {df['AQI_Category'].value_counts().to_dict()}")
    print(f"Timezone: {df['timestamp'].dt.tz} (Pakistan Time)")

    # Save to Hopsworks Feature Store
    if HOPSWORKS_ENABLED:
        print("\n" + "=" * 60)
        print("Saving features to Hopsworks Feature Store...")
        print("=" * 60)
        
        features_df = df[['timestamp', 'city', 'latitude', 'longitude',    
            'pm2_5', 'pm2_5_lag_1h', 'pm2_5_lag_24h',
            'pm10', 'pm10_lag_1h',
            'pm2_5_rolling_mean_24h', 'nitrogen_dioxide',
            'pm2_5_rolling_max_24h', 'carbon_monoxide',
            'pm10_lag_24h', 'pm10_rolling_mean_24h',
            'relative_humidity_2m', 'pm10_rolling_max_24h',
            'pm2_5_rolling_std_24h', 'sulphur_dioxide',
            'temperature_2m', 'hour_sin', 'ozone',
            'AQI', 'AQI_Category']].copy()

        if 'city' not in features_df.columns:
            features_df['city'] = os.getenv("CITY", "Rawalpindi")

        # Ensure timestamp is properly formatted
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            print("⚠️  Warning: HOPSWORKS_API_KEY not set. Skipping save to Hopsworks.")
        else:
            fg = save_features_to_hopsworks(
                features_df=features_df,
                feature_group_name="aqi_features",
                description="AQI forecasting features (Pakistan Time) with time-based and derived features"
            )
            if fg is not None:
                print("✓ Features successfully saved to Hopsworks Feature Store!")
            else:
                print("⚠️  Warning: Could not save to Hopsworks. Features are still saved locally.")

    return df


def main():
    # Find raw files (both historical and realtime)
    raw_files = list(RAW_DIR.glob("openmeteo_combined_*.csv")) + list(RAW_DIR.glob("realtime_*.csv"))
    
    if not raw_files:
        print("No raw data found in data/raw/. Run fetch script first.")
        return

    # Use the most recent file
    latest_file = max(raw_files, key=lambda f: f.stat().st_mtime)
    print(f"\n📂 Using latest file: {latest_file.name}\n")
    
    df = process_file(latest_file)

    if df is None:
        print("Error: Processing failed for the latest file. Aborting.")
        return

    # Validation
    required_cols = ["AQI", "timestamp", "pm2_5", "pm10"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Processed dataframe missing required columns: {missing}. Aborting.")
        return

    print(f"\n✅ Processing complete!")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Timezone: Pakistan Time (Asia/Karachi)")

if __name__ == "__main__":
    main()