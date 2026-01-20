import os
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pytz

# ==================== CONFIG ====================
load_dotenv()

LAT = float(os.getenv("LAT", "33.5973"))
LON = float(os.getenv("LON", "73.0479"))
CITY = os.getenv("CITY", "Rawalpindi")

# Date range for historical data (in Pakistan dates)
START_DATE = "2025-01-01"  # Pakistan date
END_DATE = "2026-01-15"    # Pakistan date

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_PARAMS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
WEATHER_PARAMS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation"]

# Pakistan timezone
PAKISTAN_TZ = pytz.timezone('Asia/Karachi')


# ==================== HELPERS ====================
def fetch_api(url, params):
    """Fetch API data and return DataFrame."""
    try:
        print(f"Fetching from {url.split('/')[-2]}...")
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        if "time" not in data:
            print("  ❌ No hourly data in response")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        
        # Convert to Pakistan timezone
        df["timestamp"] = pd.to_datetime(df["time"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(PAKISTAN_TZ)
        df.drop(columns=["time"], inplace=True)
        
        print(f"  ✓ Fetched {len(df)} records")
        return df
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return pd.DataFrame()


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print(f"HISTORICAL DATA FETCH (Pakistan Time)")
    print("=" * 70)
    print(f"Location: {CITY} ({LAT}, {LON})")
    print(f"Date range: {START_DATE} to {END_DATE} (Pakistan dates)")
    print(f"Timezone: Asia/Karachi (UTC+5)")
    print("=" * 70)

    # Air Quality Data
    print("\n🌫️  Fetching air quality data...")
    air_df = fetch_api(AIR_URL, {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(AIR_PARAMS),
        "timezone": "Asia/Karachi"  # ✅ Use Pakistan timezone
    })

    # Weather Data (using archive API for historical data)
    print("\n🌤️  Fetching weather data...")
    weather_df = fetch_api(WEATHER_URL, {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(WEATHER_PARAMS),
        "timezone": "Asia/Karachi"  # ✅ Use Pakistan timezone
    })

    if air_df.empty or weather_df.empty:
        print("\n❌ Failed to fetch data. Check your internet or API.")
        return

    # Merge air and weather data
    print("\n🔗 Merging datasets...")
    df = pd.merge(air_df, weather_df, on="timestamp", how="inner")
    
    if df.empty:
        print("❌ No data after merge")
        return
    
    print(f"✓ Merged {len(df)} records")
    
    df["city"] = CITY
    df["latitude"] = LAT
    df["longitude"] = LON

    # Drop duplicates and missing values
    before_dedup = len(df)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    print(f"✓ Removed {before_dedup - len(df)} duplicate records")
    
    before_dropna = len(df)
    df.dropna(subset=["pm2_5", "pm10"], inplace=True)
    print(f"✓ Removed {before_dropna - len(df)} records with missing PM values")

    # Save data
    filename = OUTPUT_DIR / f"openmeteo_combined_{CITY.lower()}_{START_DATE.replace('-', '')}-{END_DATE.replace('-', '')}.csv"
    df.to_csv(filename, index=False)

    print("\n" + "=" * 70)
    print("✅ HISTORICAL DATA SAVED")
    print("=" * 70)
    print(f"File: {filename}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range (PKT): {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Timezone: {df['timestamp'].dt.tz}")
    print("\n📌 Next step: Run feature extraction")
    print("   python src/features/features_extraction.py")

if __name__ == "__main__":
    main()