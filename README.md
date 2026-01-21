# Air Quality Index (AQI) Forecasting Project

Streamlit app Deployed here: https://airqualityindex-br5gsqjzqcxrgtwfrihf3a.streamlit.app/

A comprehensive end-to-end machine learning pipeline for predicting Air Quality Index (AQI) in the next 3 days using a 100% serverless stack.

## 🌟 Project Overview

This project implements a complete ML pipeline for AQI forecasting with:
- **Automated data collection** from Open-Meteo APIs
- **Feature engineering** with time-based and derived features
- **Multiple ML models** (Random Forest, LightGBM, XGBoost)
- **Real-time predictions** through a web dashboard
- **Model explainability** using SHAP
- **CI/CD automation** with GitHub Actions

## 📋 Features

### ✅ Feature Pipeline Development
- Fetch raw weather and pollutant data from Open-Meteo APIs
- Compute features including:
  - Time-based features (hour, day, month, cyclical encodings)
  - Derived features (AQI change rate, rolling statistics, lag features)
  - Interaction features
- Store processed features for model training

### ✅ Historical Data Backfill
- Run feature pipeline for past dates to generate training data
- Create comprehensive dataset for model training and evaluation

### ✅ Training Pipeline Implementation
- Fetch historical features and targets
- Experiment with multiple ML models:
  - Random Forest Regressor
  - LightGBM Regressor
  - XGBoost Regressor
- Evaluate performance using RMSE, MAE, and R² metrics
- Store trained models in Model Registry

### ✅ Automated CI/CD Pipeline
- Feature pipeline runs every hour automatically via `.github/workflows/feature_pipeline.yml` (hourly scheduling)
- Training pipeline runs daily (default 02:00 UTC) via `.github/workflows/training_pipeline.yml`
- Set `HOPSWORKS_API_KEY` in repository secrets to enable Feature Store usage in the training workflow
- The hourly feature pipeline uses a small augmentation (`BACKFILL_AUGMENT_STD=0.01`) to diversify historical training examples and help prevent model overfitting

### ✅ Web Application Dashboard
- Streamlit dashboard for real-time predictions
- Interactive visualizations with Plotly
- Display predictions for next 3 days
- Historical comparison charts

### ✅ Advanced Analytics Features
- Exploratory Data Analysis (EDA) notebooks
- SHAP for feature importance explanations
- Alerts for hazardous AQI levels
- Support for multiple forecasting models

## 🛠️ Technology Stack

- **Python 3.10+**
- **Scikit-learn** - Traditional ML models
- **XGBoost** - Gradient boosting models
- **Hopsworks** (optional) - Feature Store
- **Apache Airflow** (optional) - Workflow orchestration
- **Streamlit** - Web dashboard
- **Flask/FastAPI** - API backend
- **Open-Meteo API** - Weather and air quality data
- **SHAP** - Model explainability
- **GitHub Actions** - CI/CD automation

## 📁 Project Structure

```
Air Quality Index/
├── data/
│   ├── raw/              # Raw data from APIs
│   └── processed/        # Processed features
├── models/               # Trained models
├── metrics/              # Training metrics
├── reports/              # SHAP analysis reports
├── src/
│   ├── ingestion/        # Data fetching scripts
│   │   ├── fetchdata.py          # Historical data
│   │   └── fetch_realtime.py     # Real-time data
│   ├── features/         # Feature engineering
│   │   └── features_extraction.py
│   ├── training/         # Model training
│   │   └── train_models.py
│   ├── prediction/       # Prediction utilities
│   │   └── predict.py
│   ├── dashboard/        # Streamlit dashboard
│   │   └── app.py
│   ├── explainability/   # SHAP analysis
│   │   └── shap_analysis.py
│   └── Exploring_data/   # EDA notebooks
│       └── EDA.ipynb
├── .github/
│   └── workflows/        # CI/CD pipelines
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) Hopsworks account for Feature Store

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "Air Quality Index"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv AQI
   source AQI/bin/activate  # On Windows: AQI\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   LAT=33.7215
   LON=73.0433
   CITY=Islamabad
   ```

### Usage

#### 1. Fetch Historical Data
```bash
# Example: download historical raw data (if available)
python src/ingestion/fetchdata.py

# Backfill historical files into processed features (one-off or range)
python src/ingestion/backfill.py --start-date 2025-01-01 --end-date 2025-12-31 --augment-std 0.01

# Run feature extraction for the latest raw file
python src/features/features_extraction.py
```
python src/ingestion/fetchdata.py
```

#### 2. Extract Features
```bash
python src/features/features_extraction.py
```

#### 3. Train Models
```bash
python src/training/train_models.py
```

#### 4. Generate SHAP Analysis
```bash
python src/explainability/shap_analysis.py
```

#### 5. Run Dashboard
```bash
streamlit run src/dashboard/app.py
```

#### 6. Fetch Real-time Data
```bash
python src/ingestion/fetch_realtime.py
```

## 📊 Model Performance

Models are evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

Training metrics are saved in the `metrics/` directory.

## 🔄 CI/CD Pipeline

The project uses GitHub Actions to automate both the feature pipeline and model training pipeline.

### Feature Pipeline (Hourly)
**Workflow:** `.github/workflows/feature_pipeline.yml`

- **Schedule:** Runs every hour at minute 0 (cron: `0 * * * *`)
- **What it does:**
  - Fetches real-time air quality and weather data from Open-Meteo APIs
  - Extracts features (time-based, derived, rolling statistics)
  - Saves processed features to Hopsworks Feature Store (if configured)
  - Stores raw data locally (if not using Hopsworks)
- **Duration:** ~5-10 minutes per run
- **Manual trigger:** Can be manually triggered from GitHub Actions UI

### Training Pipeline (Daily)
**Workflow:** `.github/workflows/training_pipeline.yml`

- **Schedule:** Runs daily at 02:00 UTC (cron: `0 2 * * *`)
- **What it does:**
  - Fetches training dataset from Hopsworks Feature Store
  - Trains three ML models (Random Forest, LightGBM, XGBoost)
  - Evaluates models using RMSE, MAE, and R² metrics
  - Registers best model to Hopsworks Model Registry
  - Uploads models and metrics as GitHub Actions artifacts
- **Duration:** ~15-30 minutes per run
- **Manual trigger:** Can be manually triggered from GitHub Actions UI

### Setting Up GitHub Actions

#### 1. Configure GitHub Secrets

Go to your repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

**Required Secrets:**
- `HOPSWORKS_API_KEY` (Required for training pipeline) - Your Hopsworks API key

**Optional Secrets (with defaults):**
- `LAT` - Latitude (default: `33.5973`)
- `LON` - Longitude (default: `73.0479`)
- `CITY` - City name (default: `Rawalpindi`)
- `FEATURE_GROUP_NAME` - Hopsworks feature group name (default: `aqi_features`)
- `TRAINING_SAMPLE_FRAC` - Optional sampling fraction for training (e.g., `0.5` for 50%)
- `TRAINING_AUGMENT_STD` - Optional augmentation standard deviation (e.g., `0.01`)

#### 2. Enable GitHub Actions

1. Push the workflow files to your repository
2. Go to the **Actions** tab in your GitHub repository
3. You should see both workflows listed
4. Workflows will run automatically on schedule, or you can trigger them manually

#### 3. Monitor Workflows

- View workflow runs in the **Actions** tab
- Check logs for any errors
- Download artifacts (models, metrics) from successful training runs
- Set up notifications for workflow failures (GitHub Settings → Notifications)

#### 4. Troubleshooting

**Feature Pipeline Issues:**
- Check API connectivity (Open-Meteo APIs are public, no key needed)
- Verify environment variables are set correctly
- Check if Hopsworks API key is valid (if using Feature Store)

**Training Pipeline Issues:**
- Ensure `HOPSWORKS_API_KEY` is set (required)
- Verify feature group exists in Hopsworks
- Check training data availability in Feature Store
- Review model training logs for errors

**Common Issues:**
- If workflows don't run, check repository settings → Actions → General → Workflow permissions
- Ensure workflows are enabled (not disabled by repository settings)
- Check cron syntax if schedule doesn't work (GitHub Actions uses UTC time)

## 📈 Dashboard Features

The Streamlit dashboard provides:
- Real-time AQI predictions for next 3 days
- Interactive time series visualizations
- Current vs predicted AQI comparison
- Feature importance analysis (SHAP)
- Historical trend analysis
- Alerts for hazardous AQI levels

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) is used to:
- Understand feature importance
- Explain individual predictions
- Identify key factors affecting AQI

Reports are generated in `reports/shap/`.

## 🌐 API Integration

The project uses:
- **Open-Meteo Air Quality API** - For pollutant data
- **Open-Meteo Weather API** - For weather data

No API keys required for basic usage.

## 📝 Configuration

### Environment Variables

Create a `.env` file in the project root for local development:
```env
LAT=33.5973
LON=73.0479
CITY=Rawalpindi
HOPSWORKS_API_KEY=your_api_key_here
FEATURE_GROUP_NAME=aqi_features
TRAINING_SAMPLE_FRAC=  # Optional: sampling fraction (0.0-1.0)
TRAINING_AUGMENT_STD=  # Optional: augmentation std deviation
```

### GitHub Actions Secrets

See the [CI/CD Pipeline](#-cicd-pipeline) section above for detailed instructions on setting up GitHub Secrets.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Next Steps

- [ ] Integrate with Hopsworks Feature Store
- [ ] Add Vertex AI integration
- [ ] Implement model versioning
- [ ] Add more advanced deep learning models
- [ ] Create REST API with FastAPI
- [ ] Add unit tests
- [ ] Deploy to cloud (AWS/GCP/Azure)

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Built with ❤️ for better air quality monitoring**
