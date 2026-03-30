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
    page_icon="💨",
    layout="wide"
)

# ================= CONFIG =================
LOCAL_TZ = pytz.timezone("Asia/Karachi")
PROJECT_NAME = "AQI_Prediction_ak"
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

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global Typography & Background */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Animations */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Main container fade-in */
        .block-container {
            animation: fadeInUp 0.8s ease-out;
        }

        /* Button Enhancements */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
            border: none;
            color: white;
        }
        .stButton > button:active {
            transform: translateY(0);
        }

        /* AQI Custom Cards */
        .aqi-card {
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            color: white;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin: 10px 0;
            position: relative;
            overflow: hidden;
        }
        
        .aqi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
            opacity: 0.5;
            pointer-events: none;
        }

        .aqi-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
        }
        
        /* Metric container styling */
        [data-testid="stMetricValue"] {
            font-weight: 700;
        }
        
        /* Hazard Alerts */
        .hazard-alert {
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            color: white;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(220, 38, 38, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }
        
        /* Legend Cards */
        .legend-card {
            padding: 12px 16px;
            border-radius: 10px;
            margin: 8px 0;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        .legend-card:hover {
            transform: translateX(5px);
        }
        </style>
    """, unsafe_allow_html=True)


# AQI UI Properties
def get_aqi_ui_properties(aqi):
    if aqi <= 50:
        return "Good", "linear-gradient(135deg, #10b981, #059669)", "#ffffff"
    elif aqi <= 100:
        return "Moderate", "linear-gradient(135deg, #fbbf24, #d97706)", "#ffffff"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "linear-gradient(135deg, #f97316, #ea580c)", "#ffffff"
    elif aqi <= 200:
        return "Unhealthy", "linear-gradient(135deg, #ef4444, #dc2626)", "#ffffff"
    elif aqi <= 300:
        return "Very Unhealthy", "linear-gradient(135deg, #a855f7, #7e22ce)", "#ffffff"
    else:
        return "Hazardous", "linear-gradient(135deg, #9f1239, #881337)", "#ffffff"

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

    project = hopsworks.login(project=PROJECT_NAME, api_key_value=api_key, host="eu-west.cloud.hopsworks.ai")
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
inject_custom_css()
st.title("💨 Air Quality Index (AQI) Forecast Dashboard")
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
with st.status("Fetching latest air quality insights...", expanded=True) as status:
    st.write("🌍 Connecting to Feature Store...")
    project = load_hopsworks_connection()
    st.write("📊 Fetching recent AQI data...")
    df = load_feature_data(project)
    st.write("🤖 Loading AI prediction models...")
    best_model, best_model_name, best_rmse, metrics_df = load_all_models(project)
    st.write("🔮 Generating 72-hour forecast sequence...")
    forecast_df, daily_forecast, last_timestamp = generate_forecast(df, best_model)
    status.update(label="Forecast ready!", state="complete", expanded=False)

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
    category, background, text_color = get_aqi_ui_properties(row["predicted_aqi"])
    
    with cols[idx]:
        st.markdown(f"""
        <div class="aqi-card" style="background: {background};">
            <h3 style="margin:0; font-weight:500; font-size: 1.1rem; opacity: 0.9;">{row['date'].strftime('%a, %b %d')}</h3>
            <h1 style="margin:10px 0; font-size:3.5rem; font-weight:700; letter-spacing: -1px;">{row['predicted_aqi']:.0f}</h1>
            <p style="margin:0; font-size:1rem; font-weight:600; display: inline-block; padding: 4px 12px; background: rgba(255,255,255,0.2); border-radius: 20px;">{category}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ================= HAZARDOUS ALERTS =================
st.header("⚠️ Hazard Alerts")

hazardous_days = daily_forecast[daily_forecast["predicted_aqi"] > 200]

if len(hazardous_days) > 0:
    st.error("🚨 **HAZARDOUS AIR QUALITY DETECTED!**")
    for _, row in hazardous_days.iterrows():
        category, background, text_color = get_aqi_ui_properties(row["predicted_aqi"])
        st.markdown(f"""
        <div class="hazard-alert" style="background: {background};">
            <h3 style="margin:0; font-size: 1.5rem; font-weight: 700;">⚠️ {row['date'].strftime('%A, %B %d')}</h3>
            <p style="margin:8px 0; font-size:1.1rem; font-weight: 500;">
                Predicted AQI: <strong style="font-size:1.3rem;">{row['predicted_aqi']:.1f}</strong> - {category}
            </p>
            <p style="margin:0; font-size: 0.9rem; opacity: 0.9;">
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
fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.1, line_width=0)
fig.add_hrect(y0=50, y1=100, fillcolor="#fbbf24", opacity=0.1, line_width=0)
fig.add_hrect(y0=100, y1=150, fillcolor="#f97316", opacity=0.1, line_width=0)
fig.add_hrect(y0=150, y1=200, fillcolor="#ef4444", opacity=0.1, line_width=0)
fig.add_hrect(y0=200, y1=300, fillcolor="#a855f7", opacity=0.1, line_width=0)
fig.add_hrect(y0=300, y1=500, fillcolor="#9f1239", opacity=0.1, line_width=0)

fig.update_layout(
    xaxis_title="Date & Time",
    yaxis_title="AQI",
    hovermode='x unified',
    height=450,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=20, b=20)
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
    ("0-50", "Good", "linear-gradient(135deg, #10b981, #059669)", "Air quality is satisfactory"),
    ("51-100", "Moderate", "linear-gradient(135deg, #fbbf24, #d97706)", "Acceptable for most people"),
    ("101-150", "Unhealthy for Sensitive Groups", "linear-gradient(135deg, #f97316, #ea580c)", "May affect sensitive individuals"),
    ("151-200", "Unhealthy", "linear-gradient(135deg, #ef4444, #dc2626)", "Everyone may experience health effects"),
    ("201-300", "Very Unhealthy", "linear-gradient(135deg, #a855f7, #7e22ce)", "Health alert: everyone may experience serious effects"),
    ("301+", "Hazardous", "linear-gradient(135deg, #9f1239, #881337)", "Emergency conditions: everyone is likely affected")
]

for aqi_range, category, background, description in legend_data:
    st.markdown(f"""
    <div class="legend-card" style="background: {background};">
        <strong style="font-size:1.1rem;">{aqi_range}</strong> &nbsp;|&nbsp; 
        <span style="font-weight:600; letter-spacing:0.5px;">{category}</span><br/>
        <span style="font-size:0.9rem; opacity:0.9;">{description}</span>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(f"🕐 Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} | Data from Hopsworks Feature Store")
