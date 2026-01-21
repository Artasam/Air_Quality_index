import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pytz
import os
import hopsworks
import joblib
import tempfile
from pathlib import Path
import shutil
from dotenv import load_dotenv
from datetime import timedelta

# Page config
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# ================= CONFIG =================
LOCAL_TZ = pytz.timezone("Asia/Karachi")
PROJECT_NAME = "abrk8300"
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
CITY_NAME = "Rawalpindi"
FORECAST_HOURS = 72

FEATURE_COLUMNS = [
    'pm2_5', 'pm2_5_lag_1h', 'pm2_5_lag_24h', 'pm10', 'pm10_lag_1h',
    'pm2_5_rolling_mean_24h', 'nitrogen_dioxide', 'pm2_5_rolling_max_24h',
    'carbon_monoxide', 'pm10_lag_24h', 'pm10_rolling_mean_24h',
    'relative_humidity_2m', 'pm10_rolling_max_24h', 'pm2_5_rolling_std_24h',
    'sulphur_dioxide', 'temperature_2m', 'hour_sin', 'ozone'
]

MODEL_NAMES = ["aqi_lightgbm", "aqi_xgboost", "aqi_random_forest"]

# AQI Categories
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

# ================= CACHED FUNCTIONS =================
@st.cache_resource
def load_hopsworks_connection():
    """Load Hopsworks connection (cached)"""
           # Try Streamlit secrets first (for cloud), then .env (for local)
    try:
        api_key = st.secrets["HOPSWORKS_API_KEY"]
    except:
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not api_key:
        st.error("❌ HOPSWORKS_API_KEY not found. Please add it to Streamlit secrets.")
        st.stop()

    project = hopsworks.login(project=PROJECT_NAME, api_key_value=api_key)
    return project

@st.cache_data(ttl=3600)
def load_feature_data(_project):
    """Load feature data from Hopsworks (cached for 1 hour)"""
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
    df = (
        fg.read()
          .query("city == @CITY_NAME")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    
    if len(df) < 24:
        st.error("Need at least 24 hours of data for rolling features")
        st.stop()
    
    return df

@st.cache_resource
def load_all_models(_project):
    """Load all models and their metrics"""
    mr = _project.get_model_registry()
    loaded_models = []
    model_metrics = []
    
    MODEL_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "aqi_models"
    
    if MODEL_DOWNLOAD_DIR.exists():
        shutil.rmtree(MODEL_DOWNLOAD_DIR)
    
    MODEL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name in MODEL_NAMES:
        try:
            all_models = mr.get_models(model_name)
            if not all_models:
                continue
            
            latest_model = max(all_models, key=lambda m: m.version)
            model_path = latest_model.download(str(MODEL_DOWNLOAD_DIR))
            
            model_files = list(Path(model_path).rglob("*.pkl"))
            if not model_files:
                model_files = list(Path(model_path).rglob("*.joblib"))
            
            if model_files:
                model = joblib.load(model_files[0])
            else:
                model = joblib.load(model_path)
            
            # Extract metrics
            metrics = latest_model.training_metrics
            rmse = metrics.get("rmse", np.nan)
            mae = metrics.get("mae", np.nan)
            r2 = metrics.get("r2", np.nan)
            
            loaded_models.append((model_name, model, rmse))
            model_metrics.append({
                "Model": model_name.replace("aqi_", "").replace("_", " ").title(),
                "RMSE": f"{rmse:.4f}" if not np.isnan(rmse) else "N/A",
                "MAE": f"{mae:.4f}" if not np.isnan(mae) else "N/A",
                "R² Score": f"{r2:.4f}" if not np.isnan(r2) else "N/A"
            })
            
        except Exception as e:
            st.warning(f"Could not load {model_name}: {e}")
            continue
    
    if not loaded_models:
        st.error("❌ No models loaded from registry")
        st.stop()
    
    best_model_name, best_model, best_rmse = min(loaded_models, key=lambda x: x[2])
    
    return best_model, best_model_name, best_rmse, pd.DataFrame(model_metrics)

def generate_forecast(df, model):
    """Generate 72-hour forecast"""
    history = df.copy().reset_index(drop=True)
    last_timestamp = history.iloc[-1]["timestamp"]
    
    # Calculate feature trends
    recent_window = history.tail(24)
    feature_trends = {}
    
    for col in FEATURE_COLUMNS:
        if col == "hour_sin":
            continue
        values = recent_window[col].values
        feature_trends[col] = (values[-1] - values[0]) / len(values)
    
    future_rows = []
    
    for step in range(1, FORECAST_HOURS + 1):
        future_time = last_timestamp + timedelta(hours=step)
        base_row = history.iloc[-1].copy()
        base_row["timestamp"] = future_time
        base_row["hour_sin"] = np.sin(2 * np.pi * future_time.hour / 24)
        
        for col in FEATURE_COLUMNS:
            if col == "hour_sin":
                continue
            base_value = history.iloc[-1][col]
            trend = feature_trends.get(col, 0)
            noise = np.random.uniform(-0.05, 0.05) * base_value
            base_row[col] = max(0, base_value + trend * step + noise)
        
        X = pd.DataFrame([base_row[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS)
        X = X.astype(float)
        predicted_aqi = model.predict(X)[0]
        
        future_rows.append({
            "timestamp": future_time,
            "predicted_aqi": predicted_aqi
        })
    
    forecast_df = pd.DataFrame(future_rows)
    forecast_df["timestamp_local"] = forecast_df["timestamp"].dt.tz_convert(LOCAL_TZ)
    forecast_df["date"] = forecast_df["timestamp_local"].dt.date
    
    current_local_date = datetime.now(LOCAL_TZ).date()
    forecast_df_filtered = forecast_df[forecast_df["date"] > current_local_date].copy()
    
    daily_forecast = (
        forecast_df_filtered.groupby("date")["predicted_aqi"]
        .mean()
        .reset_index()
        .head(3)
    )
    
    return forecast_df, daily_forecast, last_timestamp

# ================= MAIN APP =================
st.title("🌡️ Air Quality Index (AQI) Forecast Dashboard")
st.markdown(f"### 📍 {CITY_NAME} - Next 3 Days Prediction")

# Sidebar
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.info("💡 Data refreshes automatically every hour")
    st.markdown(f"**Location:** {CITY_NAME}")
    st.markdown(f"**Timezone:** {LOCAL_TZ}")

# Load data and models
with st.spinner("Loading data and models..."):
    project = load_hopsworks_connection()
    df = load_feature_data(project)
    best_model, best_model_name, best_rmse, metrics_df = load_all_models(project)
    forecast_df, daily_forecast, last_timestamp = generate_forecast(df, best_model)

# Data status
current_time = datetime.now(LOCAL_TZ)
data_age_hours = (current_time - last_timestamp.astimezone(LOCAL_TZ)).total_seconds() / 3600

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Last Data Update", last_timestamp.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M"))
with col2:
    st.metric("⏰ Data Age", f"{abs(data_age_hours):.1f} hours")
with col3:
    st.metric("🤖 Best Model", best_model_name.replace("aqi_", "").title())

# Warning if data is old
if data_age_hours > 24:
    st.warning(f"⚠️ Data is {data_age_hours/24:.1f} days old. Consider refreshing the pipeline.")

st.markdown("---")

# ================= DAILY FORECAST =================
st.header("📅 3-Day AQI Forecast")

cols = st.columns(3)
for idx, (_, row) in enumerate(daily_forecast.iterrows()):
    category, color = get_aqi_category(row["predicted_aqi"])
    
    with cols[idx]:
        st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h3 style="color:white; margin:0;">{row['date'].strftime('%a, %b %d')}</h3>
            <h1 style="color:white; margin:10px 0; font-size:48px;">{row['predicted_aqi']:.0f}</h1>
            <p style="color:white; margin:0; font-size:18px;"><strong>{category}</strong></p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ================= HAZARDOUS ALERTS =================
st.header("⚠️ Hazard Alerts")

hazardous_days = daily_forecast[daily_forecast["predicted_aqi"] > 200]

if len(hazardous_days) > 0:
    st.error("🚨 **HAZARDOUS AIR QUALITY DETECTED!**")
    for _, row in hazardous_days.iterrows():
        category, color = get_aqi_category(row["predicted_aqi"])
        st.markdown(f"""
        <div style="background-color:{color}; padding:15px; border-radius:5px; margin:10px 0;">
            <h3 style="color:white; margin:0;">⚠️ {row['date'].strftime('%A, %B %d')}</h3>
            <p style="color:white; margin:5px 0; font-size:18px;">
                Predicted AQI: <strong>{row['predicted_aqi']:.1f}</strong> - {category}
            </p>
            <p style="color:white; margin:0;">
                ⛔ Stay indoors • 😷 Wear N95 mask if going out • 🏠 Use air purifiers
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.success("✅ No hazardous air quality alerts for the next 3 days")

st.markdown("---")

# ================= HOURLY FORECAST CHART =================
st.header("📈 Hourly Forecast (72 Hours)")

fig = go.Figure()

# Add hourly predictions
fig.add_trace(go.Scatter(
    x=forecast_df["timestamp_local"],
    y=forecast_df["predicted_aqi"],
    mode='lines+markers',
    name='Predicted AQI',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=4)
))

# Add AQI category zones
fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
fig.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, line_width=0)

fig.update_layout(
    xaxis_title="Date & Time",
    yaxis_title="AQI",
    hovermode='x unified',
    height=400,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ================= MODEL METRICS =================
st.header("🎯 Model Performance Metrics")

st.dataframe(
    metrics_df,
    use_container_width=True,
    hide_index=True
)

st.caption(f"**Best Model:** {best_model_name.replace('aqi_', '').title()} (RMSE: {best_rmse:.4f})")

st.markdown("---")

# ================= AQI LEGEND =================
st.header("📊 AQI Category Reference")

legend_data = [
    ("0-50", "Good", "#00e400", "Air quality is satisfactory"),
    ("51-100", "Moderate", "#ffff00", "Acceptable for most people"),
    ("101-150", "Unhealthy for Sensitive Groups", "#ff7e00", "May affect sensitive individuals"),
    ("151-200", "Unhealthy", "#ff0000", "Everyone may experience health effects"),
    ("201-300", "Very Unhealthy", "#8f3f97", "Health alert: everyone may experience serious effects"),
    ("301+", "Hazardous", "#7e0023", "Emergency conditions: everyone is likely affected")
]

for aqi_range, category, color, description in legend_data:
    st.markdown(f"""
    <div style="background-color:{color}; padding:10px; border-radius:5px; margin:5px 0;">
        <strong style="color:white;">{aqi_range}</strong> - 
        <span style="color:white;">{category}</span>: 
        <span style="color:white; font-size:14px;">{description}</span>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(f"🕐 Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} | Data from Hopsworks Feature Store")