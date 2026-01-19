import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import os

# ==================== CONFIG ====================
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Model name mapping to Hopsworks model names
HOPSWORKS_MODEL_NAMES = {
    "random_forest": "aqi_random_forest",
    "lightgbm": "aqi_lightgbm",
    "xgboost": "aqi_xgboost"
}


def load_model(model_name="random_forest", use_hopsworks=True):
    """
    Load a trained model, attempting Hopsworks first, then falling back to local.
    
    Args:
        model_name: Model name ("random_forest", "lightgbm", "xgboost")
        use_hopsworks: Whether to try loading from Hopsworks first (default: True)
    
    Returns:
        Tuple of (model, model_type, scaler)
    """
    model_type = model_name
    
    # Try loading from Hopsworks first if enabled
    if use_hopsworks:
        try:
            from src.feature_store.hopsworks_integration import load_model_from_hopsworks
            
            # Determine model type for Hopsworks (all our models are sklearn)
            hopsworks_name = HOPSWORKS_MODEL_NAMES.get(model_name)
            if hopsworks_name:
                model, version, model_path = load_model_from_hopsworks(
                    model_name=hopsworks_name,
                    model_type="sklearn"
                )
                if model is not None:
                    print(f"✓ Loaded {model_name} model from Hopsworks (v{version})")
                    return model, model_type, None
        except Exception as e:
            print(f"Note: Could not load model from Hopsworks: {e}")
            print("   Falling back to local model...")
    
    # Fallback to local model
    if model_name == "random_forest":
        model_path = MODELS_DIR / "random_forest_model.pkl"
    elif model_name == "lightgbm":
        model_path = MODELS_DIR / "lightgbm_model.pkl"
    elif model_name == "xgboost":
        model_path = MODELS_DIR / "xgboost_model.pkl"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please train models first or ensure Hopsworks is configured."
        )
    
    model = joblib.load(model_path)
    print(f"✓ Loaded {model_name} model from local storage")
    return model, model_type, None


def load_model_for_shap(model_name="random_forest", use_hopsworks=True):
    """Load model specifically for SHAP analysis. Returns (model, model_type)."""
    model, model_type, _ = load_model(model_name, use_hopsworks=use_hopsworks)
    return model, model_type


def load_feature_info():
    """Load feature column information."""
    feature_info_path = MODELS_DIR / "feature_info.json"
    if not feature_info_path.exists():
        raise FileNotFoundError(f"Feature info not found at {feature_info_path}")
    
    with open(feature_info_path, 'r') as f:
        return json.load(f)


def prepare_features(df, feature_cols):
    """Prepare features for prediction."""
    # Ensure all required features are present
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        # Add missing columns with default values
        for col in missing_cols:
            df[col] = 0
    
    # Select and order features
    X = df[feature_cols].fillna(0)
    return X


def predict_aqi(df, model_name="random_forest", use_hopsworks=True):
    """
    Predict AQI for given features.
    
    Args:
        df: DataFrame with features
        model_name: Model name to use
        use_hopsworks: Whether to try loading model from Hopsworks first
    
    Returns:
        Array of predictions
    """
    # Load model and feature info
    model, model_type, scaler = load_model(model_name, use_hopsworks=use_hopsworks)
    
    # Try to load feature info from Hopsworks metadata or local file
    try:
        feature_info = load_feature_info()
        feature_cols = feature_info['feature_columns']
    except FileNotFoundError:
        # If feature_info.json not found, try to infer from model or use all numeric columns
        print("Warning: feature_info.json not found. Inferring features from data.")
        exclude_cols = ['AQI', 'AQI_Category', 'timestamp', 'city', 'latitude', 'longitude', 
                       'current_aqi', 'aqi_category', 'predicted_aqi', 'predicted_category']
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        print(f"Using {len(feature_cols)} inferred features")
    
    # Prepare features
    X = prepare_features(df, feature_cols)
    X = X.values
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions


def predict_next_3_days(realtime_data, model_name="random_forest"):
    """Predict AQI for next 3 days from real-time data."""
    predictions = predict_aqi(realtime_data, model_name)
    
    # Add predictions to dataframe
    result_df = realtime_data.copy()
    result_df['predicted_aqi'] = predictions
    
    return result_df
