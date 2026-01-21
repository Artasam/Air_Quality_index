import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import os

# -----------------------------
# CI-SAFE HOPSWORKS SETTINGS
# -----------------------------
os.environ.setdefault("HOPSWORKS_UPLOAD_CONCURRENCY", "1")
os.environ.setdefault("HOPSWORKS_CLIENT_MAX_RETRIES", "5")
os.environ.setdefault("HOPSWORKS_CLIENT_RETRY_BACKOFF", "2")

# Ensure project root is on sys.path so `src` imports work when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load environment variables from .env if present
from dotenv import load_dotenv
load_dotenv()

# ==================== CONFIG ====================
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path(__file__).resolve().parents[2] / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==================== METRICS ====================
def calculate_metrics(y_true, y_pred, model_name):
    """Calculate and return evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "model": model_name,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n{model_name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return metrics


# ==================== MODELS ====================
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    print("\nTraining Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, "RandomForest")
    
    # Save model
    model_path = MODELS_DIR / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")

    # Register model to Hopsworks model registry (MANDATORY)
    from src.feature_store.hopsworks_integration import save_model_to_hopsworks
    ok = save_model_to_hopsworks(
        model_path, 
        model_name="aqi_random_forest",
        description="RandomForest model for AQI forecasting",
        metadata=metrics
    )
    if not ok:
        raise RuntimeError("Failed to upload RandomForest model to Hopsworks model registry. Aborting as Hopsworks model registry is mandatory.")
    print("✓ Model registered to Hopsworks model registry")

    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM model."""
    print("\nTraining LightGBM...")
    
    model = lgb.LGBMRegressor(
        
        random_state=RANDOM_STATE
       
    )
    
    # Handle different LightGBM versions for early stopping
    try:
        # Try with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        )
    except (TypeError, AttributeError, ValueError):
        # Fallback: Train without early stopping
        print("  Note: Training without early stopping...")
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, "LightGBM")
    
    # Save model
    model_path = MODELS_DIR / "lightgbm_model.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")

    # Register model to Hopsworks model registry (MANDATORY)
    from src.feature_store.hopsworks_integration import save_model_to_hopsworks
    ok = save_model_to_hopsworks(
        model_path, 
        model_name="aqi_lightgbm",
        description="LightGBM model for AQI forecasting",
        metadata=metrics
    )
    if not ok:
        raise RuntimeError("Failed to upload LightGBM model to Hopsworks model registry. Aborting as Hopsworks model registry is mandatory.")
    print("✓ Model registered to Hopsworks model registry")

    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    print("\nTraining XGBoost...")
    
    model = xgb.XGBRegressor(random_state=RANDOM_STATE)

    # Handle different XGBoost versions for early stopping
    try:
        # Try XGBoost 2.0+ callback API
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[xgb.callback.EarlyStopping(rounds=10, save_best=True)],
            verbose=False
        )
    except (AttributeError, TypeError, ValueError) as e:
        # Fallback: Try older API with early_stopping_rounds parameter
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
        except (TypeError, ValueError) as e2:
            # Final fallback: Train without early stopping (for older versions)
            print(f"  Note: Early stopping not available, training without it...")
            model.fit(X_train, y_train, verbose=False)
    
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, "XGBoost")
    
    # Save model
    model_path = MODELS_DIR / "xgboost_model.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")

    # Register model to Hopsworks model registry (MANDATORY)
    from src.feature_store.hopsworks_integration import save_model_to_hopsworks
    ok = save_model_to_hopsworks(
        model_path, 
        model_name="aqi_xgboost",
        description="XGBoost model for AQI forecasting",
        metadata=metrics
    )
    if not ok:
        raise RuntimeError("Failed to upload XGBoost model to Hopsworks model registry. Aborting as Hopsworks model registry is mandatory.")
    print("✓ Model registered to Hopsworks model registry")

    return model, metrics


# ==================== MAIN ====================
def main():
    """Main training pipeline."""
    print("=" * 60)
    print("AQI Model Training Pipeline")
    print("=" * 60)
    
    # Load training data FROM HOPSWORKS FEATURE STORE (MANDATORY per company policy)
    df = None
    # Enforce presence of Hopsworks API key
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("Error: HOPSWORKS_API_KEY is required. Company policy mandates using Hopsworks for the Feature Store and Model Registry.")
        print("Set the HOPSWORKS_API_KEY environment variable and re-run the training.")
        return

    try:
        from src.feature_store.hopsworks_integration import create_training_dataset
        print("Fetching training dataset from Hopsworks Feature Store...")

        # Allow optional sampling and augmentation via env vars
        sample_frac = os.getenv("TRAINING_SAMPLE_FRAC")
        sample_frac = float(sample_frac) if sample_frac else None
        augment_std = float(os.getenv("TRAINING_AUGMENT_STD", "0.0"))

        fg_name = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
        df = create_training_dataset(feature_group_name=fg_name)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Optional sampling
        if sample_frac and 0 < sample_frac < 1:
            df = df.sample(frac=sample_frac, random_state=RANDOM_STATE)

        # Optional simple augmentation (adds gaussian noise to numeric features)
        if augment_std and augment_std > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                noise = pd.DataFrame(
                    np.random.normal(0, augment_std, size=df[numeric_cols].shape),
                    columns=numeric_cols,
                    index=df.index
                )
                df[numeric_cols] = df[numeric_cols] + noise

        print(f"Loaded training dataset from feature store (rows={len(df)})")
    except Exception as e:
        print(f"Error: Failed to fetch training dataset from Hopsworks: {e}")
        # Detect likely Binder/DuckDB schema errors and give actionable next steps
        err_txt = str(e).lower()
        if ("binder error" in err_txt) or ("referenced column" in err_txt) or ("{'name':" in str(e)):
            print("It looks like the Hopsworks Query Service returned a Binder/DuckDB error (malformed feature descriptors).")
            print("Action: Run the schema inspection tool to collect diagnostics:")
            print("  python src/feature_store/hopsworks_integration.py inspect --fg <FEATURE_GROUP_NAME> --sample")
            print("If schema issues are found, consider recreating the feature group or contacting your Hopsworks administrator.")
        print("Aborting training because Hopsworks is mandatory for data and model registry.")
        if isinstance(e, ModuleNotFoundError) and "src" in str(e):
            print("Tip: The 'src' package is not importable. Ensure you run this script from the repository root or that the project root is on PYTHONPATH.")
            print("      Running 'python src/training/train_models.py' from the repository root or adding the project root to PYTHONPATH should fix this.")

    # Defensive: abort if dataset fetch failed and df is None
    if df is None:
        print("Error: No training data available. Aborting training to comply with Hopsworks-only policy.")
        return

    exclude_cols = [
    'AQI', 'aqi',
    'AQI_Category', 'aqi_category',
    'timestamp',
    'city',
    'latitude',
    'longitude'
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]


    X = df[feature_cols].fillna(0)  # Fill any remaining NaN
    y = df['AQI']
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    
    # Time-series split (preserve temporal order)
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]  # Use last split
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    all_metrics = []
    
    # 1. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    all_metrics.append(rf_metrics)
    
    # 2. LightGBM
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    all_metrics.append(lgb_metrics)
    
    # 3. XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    all_metrics.append(xgb_metrics)
    
    # Save feature columns for inference
    feature_info = {
        "feature_columns": feature_cols,
        "target_column": "AQI",
        "timestamp": datetime.now().isoformat()
    }
    feature_info_path = MODELS_DIR / "feature_info.json"
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"\n✓ Feature info saved to: {feature_info_path}")
    
    # Save all metrics
    metrics_path = METRICS_DIR / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Find best model
    best_model = min(all_metrics, key=lambda x: x['rmse'])
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_model['model']} (RMSE: {best_model['rmse']:.4f}, R²: {best_model['r2']:.4f})")
    print(f"{'=' * 60}")

    # Store best model into a simple Model Registry (models/model_registry)
    MODEL_REGISTRY_DIR = MODELS_DIR / "model_registry"
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    model_map = {
        "RandomForest": "random_forest_model.pkl",
        "LightGBM": "lightgbm_model.pkl",
        "XGBoost": "xgboost_model.pkl"
    }
    model_filename = model_map.get(best_model['model'])
    if model_filename:
        src_path = MODELS_DIR / model_filename
        if src_path.exists():
            dst_path = MODEL_REGISTRY_DIR / f"best_{best_model['model'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"✓ Best model archived to model registry: {dst_path}")

            # Register the best model in Hopsworks Model Registry (MANDATORY)
            from src.feature_store.hopsworks_integration import save_model_to_hopsworks
            ok = save_model_to_hopsworks(
                dst_path,
                model_name=f"aqi_best_{best_model['model'].lower()}",
                description=f"Best model from train run ({best_model['model']})",
                metadata=best_model
            )
            if not ok:
                raise RuntimeError("Failed to upload best model to Hopsworks model registry. Aborting as Hopsworks model registry is mandatory.")
            print("✓ Best model registered to Hopsworks model registry")
        else:
            print(f"Warning: expected model file not found: {src_path}")
    else:
        print("Warning: Best model name not recognized for registry storage.")


if __name__ == "__main__":
    main()
