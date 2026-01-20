import os
import hopsworks
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
import tempfile
from dotenv import load_dotenv
from pathlib import Path

# ================= LOAD ENV =================
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise EnvironmentError("❌ HOPSWORKS_API_KEY not found in .env file")

# ================= CONFIG =================
PROJECT_NAME = "abrk8300"
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
CITY_NAME = "Rawalpindi"

FORECAST_HOURS = 72  # 3 days
TARGET_COL = "aqi"

FEATURE_COLUMNS = [
    'pm2_5',
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
    'temperature_2m',
    'hour_sin',
    'ozone'
]

MODEL_NAMES = [
    "aqi_lightgbm",
    "aqi_xgboost",
    "aqi_random_forest"
]

# ================= LOGIN =================
project = hopsworks.login(
    project=PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()
mr = project.get_model_registry()

# ================= LOAD FEATURE DATA =================
fg = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

df = (
    fg.read()
      .query("city == @CITY_NAME")
      .sort_values("timestamp")
      .reset_index(drop=True)
)

if len(df) < 24:
    raise ValueError("Need at least 24 hours of data for rolling features")

# ================= LOAD BEST MODEL =================
loaded_models = []

# Use temp directory or user's home directory (both have write permissions)
MODEL_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "aqi_models"

# Create directory with proper error handling
try:
    MODEL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Using model directory: {MODEL_DOWNLOAD_DIR}")
except Exception as e:
    print(f"⚠️ Could not create {MODEL_DOWNLOAD_DIR}, using temp directory")
    MODEL_DOWNLOAD_DIR = Path(tempfile.mkdtemp(prefix="aqi_models_"))
    print(f"📁 Using temporary directory: {MODEL_DOWNLOAD_DIR}")

for model_name in MODEL_NAMES:
    try:
        print(f"\n🔄 Attempting to load {model_name}...")
        
        # Get all versions of this model
        all_models = mr.get_models(model_name)
        
        if not all_models:
            print(f"⚠️ No models found with name {model_name}")
            continue
        
        # Sort by version to get the latest
        latest_model = max(all_models, key=lambda m: m.version)
        
        print(f"   Found version {latest_model.version} (latest)")
        
        # download to temp folder
        model_path = latest_model.download(str(MODEL_DOWNLOAD_DIR))
        
        # Find the actual model file (handle nested directories)
        model_files = list(Path(model_path).rglob("*.pkl"))
        if not model_files:
            model_files = list(Path(model_path).rglob("*.joblib"))
        
        if model_files:
            actual_model_path = model_files[0]
            model = joblib.load(actual_model_path)
        else:
            # If no nested files, assume model_path is the file itself
            model = joblib.load(model_path)

        rmse = latest_model.training_metrics.get("rmse", np.inf)
        loaded_models.append((model_name, model, rmse))

        print(f"✅ Loaded {model_name} v{latest_model.version} | RMSE: {rmse:.4f}")

    except Exception as e:
        print(f"❌ Skipping {model_name}: {e}")
        continue

if not loaded_models:
    raise RuntimeError("❌ No models loaded from registry. Check your network connection and model registry.")

best_model_name, best_model, best_rmse = min(
    loaded_models, key=lambda x: x[2]
)

print(f"\n✅ Best model selected: {best_model_name} (RMSE={best_rmse:.4f})")



# ================= PREPARE TREND BASE =================
history = df.copy().reset_index(drop=True)
last_timestamp = history.iloc[-1]["timestamp"]

# Check if data is current
from datetime import datetime
import pytz

# Make current_date timezone-aware to match last_timestamp
if last_timestamp.tzinfo is not None:
    # last_timestamp is timezone-aware, make current_date aware too
    current_date = datetime.now(last_timestamp.tzinfo)
else:
    # last_timestamp is timezone-naive
    current_date = datetime.now()

data_age_hours = (current_date - last_timestamp).total_seconds() / 3600

print(f"\n📊 Data Status:")
print(f"   Last data point: {last_timestamp}")
print(f"   Current time: {current_date}")
print(f"   Data age: {data_age_hours:.1f} hours old")

if data_age_hours > 24:
    print(f"   ⚠️ WARNING: Data is {data_age_hours/24:.1f} days old!")
    print(f"   Consider running data pipeline to fetch latest data.")

recent_window = history.tail(24)
feature_trends = {}

for col in FEATURE_COLUMNS:
    if col == "hour_sin":
        continue
    values = recent_window[col].values
    feature_trends[col] = (values[-1] - values[0]) / len(values)

future_rows = []

# ================= FORECAST LOOP =================
print(f"\n🔮 Generating {FORECAST_HOURS}-hour forecast from {last_timestamp}...")

for step in range(1, FORECAST_HOURS + 1):

    future_time = last_timestamp + timedelta(hours=step)

    base_row = history.iloc[-1].copy()
    base_row["timestamp"] = future_time
    base_row["city"] = CITY_NAME

    # cyclical time
    base_row["hour_sin"] = np.sin(2 * np.pi * future_time.hour / 24)

    # synthetic features
    for col in FEATURE_COLUMNS:
        if col == "hour_sin":
            continue

        base_value = history.iloc[-1][col]
        trend = feature_trends.get(col, 0)

        noise = np.random.uniform(-0.05, 0.05) * base_value
        base_row[col] = max(0, base_value + trend * step + noise)

    # Use DataFrame instead of numpy array to preserve feature names
    X = base_row[FEATURE_COLUMNS].to_frame().T
    predicted_aqi = best_model.predict(X)[0]

    future_rows.append({
        "timestamp": future_time,
        "city": CITY_NAME,
        "predicted_aqi": predicted_aqi
    })

# ================= OUTPUT =================
forecast_df = pd.DataFrame(future_rows)
forecast_df["date"] = forecast_df["timestamp"].dt.date

daily_forecast = (
    forecast_df.groupby("date")["predicted_aqi"]
    .mean()
    .reset_index()
)

print("\n📅 Daily AQI Forecast (Next 3 Days)")
print(daily_forecast.to_string(index=False))

# Save to current working directory (usually has write permissions)
output_dir = Path.cwd()
hourly_path = output_dir / "hourly_3day_aqi_forecast.csv"
daily_path = output_dir / "daily_3day_aqi_forecast.csv"

forecast_df.to_csv(hourly_path, index=False)
daily_forecast.to_csv(daily_path, index=False)

print(f"\n✅ Forecasts saved:")
print(f"   📄 Hourly: {hourly_path}")
print(f"   📄 Daily: {daily_path}")