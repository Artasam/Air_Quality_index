import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import sys
# Add parent directory to path for feature store integration
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Add project root for feature extraction imports
try:
    from src.feature_store.hopsworks_integration import save_features_to_hopsworks
    HOPSWORKS_ENABLED = True
except ImportError:
    HOPSWORKS_ENABLED = False

# Import feature extraction functions
try:
    from src.features.features_extraction import (
        create_time_features,
        create_derived_features,
        compute_overall_aqi,
        categorize_aqi
    )
    FEATURES_EXTRACTION_ENABLED = True
except ImportError as e:
    FEATURES_EXTRACTION_ENABLED = False
    print(f"Warning: Could not import feature extraction functions: {e}")

# ==================== CONFIG ====================
load_dotenv()

LAT = float(os.getenv("LAT", "33.5973"))
LON = float(os.getenv("LON", "73.0479"))
CITY = os.getenv("CITY", "Rawalpindi")

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AIR_PARAMS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
WEATHER_PARAMS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation"]


# ==================== HELPERS ====================
def fetch_realtime_air_quality(url, params):
    """Fetch real-time air quality data."""
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        if "time" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["time"])
        df.drop(columns=["time"], inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching air quality data: {e}")
        return pd.DataFrame()


def fetch_realtime_weather(url, params):
    """Fetch real-time weather forecast data."""
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        if "time" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["time"])
        df.drop(columns=["time"], inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame()


# ==================== MAIN ====================
def main():
    """Fetch real-time air quality and weather data for current time and next 3 days."""
    print(f"Fetching real-time AQI and weather data for {CITY} ({LAT}, {LON})")
    
    # Get current time and next 3 days
    now = datetime.now()
    end_date = now + timedelta(days=3)
    
    # Air Quality Data (current + forecast)
    air_df = fetch_realtime_air_quality(AIR_URL, {
        "latitude": LAT,
        "longitude": LON,
        "start_date": now.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(AIR_PARAMS),
        "timezone": "UTC"
    })
    
    # Weather Data (current + forecast)
    weather_df = fetch_realtime_weather(WEATHER_URL, {
        "latitude": LAT,
        "longitude": LON,
        "start_date": now.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(WEATHER_PARAMS),
        "timezone": "UTC"
    })
    
    if air_df.empty or weather_df.empty:
        print("Failed to fetch real-time data. Check your internet or API.")
        return None
    
    # Merge air and weather data
    df = pd.merge(air_df, weather_df, on="timestamp", how="inner")
    df["city"] = CITY
    df["latitude"] = LAT
    df["longitude"] = LON
    
    # Drop duplicates and missing values
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.dropna(subset=["pm2_5", "pm10"], inplace=True)
    
    # Save raw data with timestamp in filename
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"realtime_{CITY.lower()}_{timestamp_str}.csv"
    df.to_csv(filename, index=False)
    
    print(f"✓ Real-time data saved to: {filename}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Extract features and save to Hopsworks Feature Store (if enabled)
    if FEATURES_EXTRACTION_ENABLED:
        try:
            print("\n" + "=" * 60)
            print("Extracting features from real-time data...")
            print("=" * 60)
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Compute AQI and Category
            print("Computing AQI...")
            df["AQI"] = df.apply(compute_overall_aqi, axis=1)
            df["AQI_Category"] = df["AQI"].apply(categorize_aqi)
            
            # Drop rows without AQI
            df.dropna(subset=["AQI"], inplace=True)
            
            if len(df) == 0:
                print("Warning: No valid rows after AQI computation. Skipping feature extraction.")
                return df
            
            # Create time-based features
            print("Creating time-based features...")
            df = create_time_features(df)
            
            # Create derived features
            print("Creating derived features...")
            df = create_derived_features(df)
            
            print(f"✓ Feature extraction complete. Features: {len(df.columns)} columns")
            
            # Prepare features DataFrame for Hopsworks (same columns as in features_extraction.py)
            # List of desired feature columns
            desired_cols = [
                'timestamp', 'city', 'latitude', 'longitude',    
                'pm2_5',    # strongest positive correlation
                'pm2_5_lag_1h',
                'pm2_5_lag_24h',
                'pm10',
                'pm10_lag_1h',
                'pm2_5_rolling_mean_24h',
                'nitrogen_dioxide',
                'pm2_5_rolling_max_24h',
                'carbon_monoxide',
                'pm10_lag_24h',
                'pm10_rolling_mean_24h',
                'relative_humidity_2m',
                'pm10_rolling_max_24h',
                'pm2_5_rolling_std_24h',
                'sulphur_dioxide',
                'temperature_2m',                # moderately negatively correlated
                'hour_sin',                       # cyclical time feature
                'ozone',                           # negative correlation
                'AQI',
                'AQI_Category'
            ]
            
            # Filter to only columns that exist in df
            available_cols = [col for col in desired_cols if col in df.columns]
            missing_cols = [col for col in desired_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Warning: Some expected feature columns are missing: {missing_cols}")
            
            # Ensure required columns are present (add defaults if missing)
            if 'timestamp' not in available_cols and 'timestamp' in df.columns:
                available_cols.insert(0, 'timestamp')
            if 'city' not in available_cols:
                if 'city' not in df.columns:
                    df['city'] = CITY
                available_cols.append('city')
            if 'latitude' not in available_cols:
                if 'latitude' not in df.columns:
                    df['latitude'] = LAT
                available_cols.append('latitude')
            if 'longitude' not in available_cols:
                if 'longitude' not in df.columns:
                    df['longitude'] = LON
                available_cols.append('longitude')
            
            # Create features DataFrame with only available columns
            features_df = df[available_cols].copy()
            
            # Filter out existing records to prevent overwrites
            if HOPSWORKS_ENABLED and len(features_df) > 0:
                try:
                    # Import function to check existing features
                    from src.feature_store.hopsworks_integration import get_features_from_hopsworks
                    
                    # Get the time range of new data
                    min_timestamp = features_df['timestamp'].min()
                    max_timestamp = features_df['timestamp'].max()
                    
                    print(f"Checking for existing records between {min_timestamp} and {max_timestamp}...")
                    
                    # Query existing features for the same time range and city
                    # Use strftime format instead of isoformat - Hopsworks filter prefers 'YYYY-MM-DD HH:MM:SS'
                    try:
                        # Format timestamps as strings in format Hopsworks can parse
                        start_time_str = min_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        end_time_str = max_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        
                        existing_df = get_features_from_hopsworks(
                            feature_group_name="aqi_features",
                            start_time=start_time_str,
                            end_time=end_time_str
                        )
                        
                        if existing_df is not None and len(existing_df) > 0:
                            # Normalize timestamp column name (may be lowercase from Hopsworks)
                            existing_df['timestamp'] = pd.to_datetime(existing_df.get('timestamp', existing_df.get('Timestamp', pd.NaT)))
                            existing_df = existing_df[existing_df['timestamp'].notna()]
                            
                            if len(existing_df) > 0:
                                # Normalize city column if needed
                                if 'city' not in existing_df.columns and 'City' in existing_df.columns:
                                    existing_df['city'] = existing_df['City']
                                
                                # Create composite key for comparison
                                existing_df['_key'] = existing_df['timestamp'].astype(str) + '_' + existing_df.get('city', 'Unknown').astype(str)
                                features_df['_key'] = features_df['timestamp'].astype(str) + '_' + features_df.get('city', CITY).astype(str)
                                
                                # Filter out rows that already exist
                                before_count = len(features_df)
                                features_df = features_df[~features_df['_key'].isin(existing_df['_key'])].copy()
                                features_df = features_df.drop(columns=['_key'], errors='ignore')
                                
                                filtered_count = before_count - len(features_df)
                                if filtered_count > 0:
                                    print(f"✓ Filtered out {filtered_count} existing record(s) to prevent overwrites")
                                
                                if len(features_df) == 0:
                                    print("✓ All records already exist in feature store. Nothing to insert.")
                                    return df
                    except Exception as check_exc:
                        # If checking fails (e.g., feature store not materialized), proceed with warning
                        print(f"⚠️  Warning: Could not check for existing records: {check_exc}")
                        print("   Proceeding with insert (mode=append to prevent overwrites)...")
                        
                except ImportError:
                    # If get_features_from_hopsworks not available, proceed with append mode
                    print("Note: Could not import get_features_from_hopsworks. Using append mode to prevent overwrites.")
            
            # Save to Hopsworks Feature Store
            if HOPSWORKS_ENABLED and len(features_df) > 0:
                print("\n" + "=" * 60)
                print("Saving features to Hopsworks Feature Store...")
                print("=" * 60)
                
                fg = save_features_to_hopsworks(
                    features_df=features_df,
                    feature_group_name="aqi_features",
                    description="AQI forecasting features with time-based and derived features (realtime)",
                    mode="append"  # Explicitly use append mode to prevent overwrites
                )
                if fg is not None:
                    print("✓ Features successfully saved to Hopsworks Feature Store!")
                else:
                    print("⚠️  Warning: Could not save to Hopsworks. Features are still saved locally.")
            else:
                print("Note: Hopsworks integration not enabled. Features extracted but not saved to feature store.")
                
        except Exception as e:
            print(f"⚠️  Error during feature extraction or Hopsworks save: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with raw data only...")
    else:
        print("\nNote: Feature extraction not available. Only raw data saved.")
        if HOPSWORKS_ENABLED:
            print("Note: To save to Hopsworks, enable feature extraction by ensuring features_extraction.py is accessible.")
    
    return df


def get_latest_realtime_data():
    """Get the most recent real-time data file."""
    realtime_files = list(OUTPUT_DIR.glob("realtime_*.csv"))
    if not realtime_files:
        return None
    
    latest_file = max(realtime_files, key=lambda f: f.stat().st_mtime)
    return pd.read_csv(latest_file)


if __name__ == "__main__":
    main()
