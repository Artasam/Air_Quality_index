import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for feature store integration
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.feature_store.hopsworks_integration import save_features_to_hopsworks, get_features_from_hopsworks
    HOPSWORKS_ENABLED = True
    logger.info("✓ Hopsworks integration enabled")
except ImportError as e:
    HOPSWORKS_ENABLED = False
    logger.warning(f"Hopsworks integration not available: {e}")

# Import feature extraction functions
try:
    from src.features.features_extraction import (
        create_time_features,
        create_derived_features,
        compute_overall_aqi,
        categorize_aqi
    )
    FEATURES_EXTRACTION_ENABLED = True
    logger.info("✓ Feature extraction enabled")
except ImportError as e:
    FEATURES_EXTRACTION_ENABLED = False
    logger.warning(f"Feature extraction not available: {e}")

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

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ==================== HELPERS ====================
def fetch_with_retry(url, params, max_retries=MAX_RETRIES):
    """Fetch data with retry logic for reliability."""
    import time
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching data (attempt {attempt + 1}/{max_retries})...")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise
    return None


def fetch_air_quality(url, params):
    """Fetch air quality data with error handling."""
    try:
        data = fetch_with_retry(url, params)
        if not data or "hourly" not in data:
            logger.error("No hourly data in API response")
            return pd.DataFrame()
        
        hourly_data = data.get("hourly", {})
        if "time" not in hourly_data:
            logger.error("No time field in hourly data")
            return pd.DataFrame()
        
        df = pd.DataFrame(hourly_data)
        df["timestamp"] = pd.to_datetime(df["time"], utc=True)
        df.drop(columns=["time"], inplace=True)
        
        logger.info(f"✓ Fetched {len(df)} air quality records")
        return df
    except Exception as e:
        logger.error(f"Error fetching air quality data: {e}")
        return pd.DataFrame()


def fetch_weather(url, params):
    """Fetch weather data with error handling."""
    try:
        data = fetch_with_retry(url, params)
        if not data or "hourly" not in data:
            logger.error("No hourly data in API response")
            return pd.DataFrame()
        
        hourly_data = data.get("hourly", {})
        if "time" not in hourly_data:
            logger.error("No time field in hourly data")
            return pd.DataFrame()
        
        df = pd.DataFrame(hourly_data)
        df["timestamp"] = pd.to_datetime(df["time"], utc=True)
        df.drop(columns=["time"], inplace=True)
        
        logger.info(f"✓ Fetched {len(df)} weather records")
        return df
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return pd.DataFrame()


# ==================== MAIN ====================
def main():
    """
    Fetch ONLY current date's ACTUAL MEASURED data (hourly).
    Designed for GitHub Actions to run every hour.
    
    Fetches: Today's data from midnight to current hour (NO FORECASTS)
    """
    logger.info("=" * 70)
    logger.info(f"Starting hourly data fetch for {CITY} ({LAT}, {LON})")
    logger.info("=" * 70)
    
    # Get current time in Pakistan timezone (Asia/Karachi = UTC+5)
    import pytz
    pakistan_tz = pytz.timezone('Asia/Karachi')
    now_local = pd.Timestamp.now(tz=pakistan_tz)
    now_utc = now_local.astimezone(pytz.UTC)  # Convert to UTC for API calls
    
    logger.info(f"Current Pakistan time: {now_local}")
    logger.info(f"Current UTC time: {now_utc}")
    logger.info(f"Current hour: {now_local.hour}:00 PKT ({now_utc.hour}:00 UTC)")
    
    # Calculate today's date range based on Pakistan local time
    # Get midnight Pakistan time, then convert to UTC
    today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    today_start = today_start_local.astimezone(pytz.UTC)
    current_hour = now_utc.replace(minute=0, second=0, microsecond=0, nanosecond=0)
    
    logger.info(f"\n📅 Fetching TODAY'S HOURLY DATA (Pakistan Time)")
    logger.info(f"   Date: {now_local.date()}")
    logger.info(f"   Time range: {today_start_local.strftime('%H:%M')} to {now_local.strftime('%H:%M')} PKT")
    logger.info(f"   (UTC: {today_start.strftime('%H:%M')} to {now_utc.strftime('%H:%M')})")
    logger.info(f"   Expected records: ~{(now_utc - today_start).total_seconds() / 3600:.0f} hours")
    logger.info("   ⚠️  NO FORECAST DATA - Only actual measurements")
    
    # Air Quality Data - fetch today's data
    logger.info("\n🌫️  Fetching air quality data...")
    air_df = fetch_air_quality(AIR_URL, {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(AIR_PARAMS),
        "timezone": "UTC"
    })
    
    # Weather Data - fetch today's data
    logger.info("\n🌤️  Fetching weather data...")
    weather_df = fetch_weather(WEATHER_URL, {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(WEATHER_PARAMS),
        "timezone": "UTC"
    })
    
    # Validation: Check if data was fetched
    if air_df.empty:
        logger.error("❌ Failed to fetch air quality data")
        return None
    
    if weather_df.empty:
        logger.error("❌ Failed to fetch weather data")
        return None
    
    # CRITICAL: Filter to ONLY today's data (midnight to current hour)
    # Remove ALL future timestamps (forecasts)
    logger.info("\n🔍 Filtering to TODAY'S DATA ONLY (removing forecasts)...")
    logger.info(f"   Keeping only: {now_local.date()} (Pakistan time)")
    logger.info(f"   UTC range: {today_start.strftime('%Y-%m-%d %H:%M')} to {current_hour.strftime('%H:%M')}")
    
    before_air = len(air_df)
    before_weather = len(weather_df)
    
    # Filter: Only keep data from today AND up to current hour
    air_df = air_df[
        (air_df['timestamp'] >= today_start) & 
        (air_df['timestamp'] <= current_hour)
    ].copy()
    
    weather_df = weather_df[
        (weather_df['timestamp'] >= today_start) & 
        (weather_df['timestamp'] <= current_hour)
    ].copy()
    
    logger.info(f"   Air quality: {before_air} → {len(air_df)} records (removed {before_air - len(air_df)} forecast/other date records)")
    logger.info(f"   Weather: {before_weather} → {len(weather_df)} records (removed {before_weather - len(weather_df)} forecast/other date records)")
    
    if air_df.empty or weather_df.empty:
        logger.error("❌ No data available for today after filtering")
        return None
    
    # Verify we only have today's data (Pakistan local date)
    air_dates = air_df['timestamp'].dt.tz_convert(pakistan_tz).dt.date.unique()
    weather_dates = weather_df['timestamp'].dt.tz_convert(pakistan_tz).dt.date.unique()
    
    if len(air_dates) > 1 or len(weather_dates) > 1:
        logger.warning(f"⚠️  Warning: Data spans multiple dates (Pakistan time): Air={air_dates}, Weather={weather_dates}")
    
    logger.info(f"✓ Data confirmed for: {now_local.date()} (Pakistan time)")
    
    # Merge air and weather data
    logger.info("\n🔗 Merging datasets...")
    df = pd.merge(air_df, weather_df, on="timestamp", how="inner")
    
    if df.empty:
        logger.error("❌ No data after merge")
        return None
    
    logger.info(f"✓ Merged dataset: {len(df)} hourly records")
    logger.info(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Add metadata
    df["city"] = CITY
    df["latitude"] = LAT
    df["longitude"] = LON
    
    # Drop duplicates and missing values
    before_dedup = len(df)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    logger.info(f"✓ Removed {before_dedup - len(df)} duplicate records")
    
    before_dropna = len(df)
    df.dropna(subset=["pm2_5", "pm10"], inplace=True)
    logger.info(f"✓ Removed {before_dropna - len(df)} records with missing PM values")
    
    if len(df) == 0:
        logger.error("❌ No valid data after cleaning")
        return None
    
    # Save raw data locally
    timestamp_str = now_utc.strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"realtime_{CITY.lower()}_{timestamp_str}.csv"
    df.to_csv(filename, index=False)
    
    logger.info(f"\n💾 Raw data saved to: {filename}")
    logger.info(f"   Rows: {len(df)} hourly records")
    logger.info(f"   Columns: {len(df.columns)}")
    logger.info(f"   Date: {now_local.date()} (Pakistan time)")
    logger.info(f"   Hours: {df['timestamp'].dt.tz_convert(pakistan_tz).dt.hour.min()}:00 to {df['timestamp'].dt.tz_convert(pakistan_tz).dt.hour.max()}:00 PKT")
    
    # Extract features and save to Hopsworks
    if not FEATURES_EXTRACTION_ENABLED:
        logger.warning("⚠️  Feature extraction disabled. Only raw data saved.")
        return df
    
    try:
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE EXTRACTION")
        logger.info("=" * 70)
        
        # Ensure timestamp is datetime with UTC timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Compute AQI and Category
        logger.info("Computing AQI...")
        df["AQI"] = df.apply(compute_overall_aqi, axis=1)
        df["AQI_Category"] = df["AQI"].apply(categorize_aqi)
        
        # Drop rows without AQI
        before_aqi = len(df)
        df.dropna(subset=["AQI"], inplace=True)
        logger.info(f"✓ Removed {before_aqi - len(df)} records without valid AQI")
        
        if len(df) == 0:
            logger.error("❌ No valid rows after AQI computation")
            return None
        
        # Create time-based features
        logger.info("Creating time-based features...")
        df = create_time_features(df)
        
        # Create derived features
        logger.info("Creating derived features...")
        df = create_derived_features(df)
        
        logger.info(f"✓ Feature extraction complete. Total features: {len(df.columns)}")
        
        # Prepare features DataFrame for Hopsworks
        desired_cols = [
            'timestamp', 'city', 'latitude', 'longitude',    
            'pm2_5', 'pm2_5_lag_1h', 'pm2_5_lag_24h',
            'pm10', 'pm10_lag_1h',
            'pm2_5_rolling_mean_24h', 'nitrogen_dioxide',
            'pm2_5_rolling_max_24h', 'carbon_monoxide',
            'pm10_lag_24h', 'pm10_rolling_mean_24h',
            'relative_humidity_2m', 'pm10_rolling_max_24h',
            'pm2_5_rolling_std_24h', 'sulphur_dioxide',
            'temperature_2m', 'hour_sin', 'ozone',
            'AQI', 'AQI_Category'
        ]
        
        # Filter to only available columns
        available_cols = [col for col in desired_cols if col in df.columns]
        missing_cols = [col for col in desired_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"⚠️  Missing feature columns: {missing_cols}")
        
        # Ensure required columns
        for col, default in [('city', CITY), ('latitude', LAT), ('longitude', LON)]:
            if col not in df.columns:
                df[col] = default
                if col not in available_cols:
                    available_cols.append(col)
        
        # Create features DataFrame
        features_df = df[available_cols].copy()
        logger.info(f"✓ Prepared {len(features_df)} records with {len(available_cols)} features")
        
        # Save to Hopsworks Feature Store
        if not HOPSWORKS_ENABLED:
            logger.warning("⚠️  Hopsworks disabled. Features saved locally only.")
            return df
        
        # Check for API key
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            logger.error("❌ HOPSWORKS_API_KEY not set. Cannot save to feature store.")
            logger.info("   Set HOPSWORKS_API_KEY in environment or .env file")
            return df
        
        logger.info("\n" + "=" * 70)
        logger.info("HOPSWORKS FEATURE STORE")
        logger.info("=" * 70)
        
        # Filter out existing records to prevent duplicates
        try:
            min_timestamp = features_df['timestamp'].min()
            max_timestamp = features_df['timestamp'].max()
            
            logger.info(f"Checking for existing records ({min_timestamp} to {max_timestamp})...")
            
            start_time_str = min_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            end_time_str = max_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            existing_df = get_features_from_hopsworks(
                feature_group_name="aqi_features",
                start_time=start_time_str,
                end_time=end_time_str
            )
            
            if existing_df is not None and len(existing_df) > 0:
                # Normalize column names
                existing_df['timestamp'] = pd.to_datetime(
                    existing_df.get('timestamp', existing_df.get('Timestamp', pd.NaT))
                )
                existing_df = existing_df[existing_df['timestamp'].notna()]
                
                if len(existing_df) > 0:
                    if 'city' not in existing_df.columns and 'City' in existing_df.columns:
                        existing_df['city'] = existing_df['City']
                    
                    # Create composite keys for deduplication
                    existing_df['_key'] = (
                        existing_df['timestamp'].astype(str) + '_' + 
                        existing_df.get('city', 'Unknown').astype(str)
                    )
                    features_df['_key'] = (
                        features_df['timestamp'].astype(str) + '_' + 
                        features_df.get('city', CITY).astype(str)
                    )
                    
                    # Filter out duplicates
                    before_count = len(features_df)
                    features_df = features_df[~features_df['_key'].isin(existing_df['_key'])].copy()
                    features_df = features_df.drop(columns=['_key'], errors='ignore')
                    
                    filtered_count = before_count - len(features_df)
                    if filtered_count > 0:
                        logger.info(f"✓ Filtered out {filtered_count} existing records")
                    
                    if len(features_df) == 0:
                        logger.info("✓ All records already exist. Nothing to insert.")
                        return df
                        
        except Exception as check_exc:
            logger.warning(f"⚠️  Could not check for existing records: {check_exc}")
            logger.info("   Proceeding with append mode...")
        
        # Save to Hopsworks
        if len(features_df) > 0:
            logger.info(f"Saving {len(features_df)} new hourly records to Hopsworks...")
            
            fg = save_features_to_hopsworks(
                features_df=features_df,
                feature_group_name="aqi_features",
                description="AQI forecasting features (hourly realtime data - no forecasts)",
                mode="append"
            )
            
            if fg is not None:
                logger.info("✓ Successfully saved to Hopsworks Feature Store!")
            else:
                logger.error("❌ Failed to save to Hopsworks")
        else:
            logger.info("No new records to save")
            
    except Exception as e:
        logger.error(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Raw data was saved successfully despite feature extraction error")
    
    logger.info("\n" + "=" * 70)
    logger.info("HOURLY DATA FETCH COMPLETE")
    logger.info("=" * 70)
    logger.info(f"✓ Collected {len(df)} hourly records for {now_local.date()} (Pakistan time)")
    logger.info(f"✓ Current hour: {now_local.hour}:00 PKT ({now_utc.hour}:00 UTC)")
    logger.info("✓ No forecast data included - actual measurements only")
    
    return df


if __name__ == "__main__":
    try:
        result = main()
        if result is not None:
            logger.info("✅ Script completed successfully")
            exit(0)
        else:
            logger.error("❌ Script failed")
            exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)