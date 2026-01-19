import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
from lime.lime_tabular import LimeTabularExplainer

# ==================== CONFIG ====================
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_features(model_name="random_forest"):
    """Load model and feature information."""
    # Load feature info
    feature_info_path = MODELS_DIR / "feature_info.json"
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)
    
    # Load model
    if model_name == "random_forest":
        model_path = MODELS_DIR / "random_forest_model.pkl"
        model = joblib.load(model_path)
        model_type = "tree"
    elif model_name == "lightgbm":
        model_path = MODELS_DIR / "lightgbm_model.pkl"
        model = joblib.load(model_path)
        model_type = "tree"
    elif model_name == "xgboost":
        model_path = MODELS_DIR / "xgboost_model.pkl"
        model = joblib.load(model_path)
        model_type = "tree"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, model_type, feature_info['feature_columns']


def generate_shap_analysis(X, model, model_type, feature_names, output_prefix="shap"):
    """Generate SHAP analysis for a model."""
    print(f"Generating SHAP analysis for {model_type} model...")
    
    # Sample data if too large (SHAP can be slow)
    if len(X) > 100:
        X_sample = X.sample(n=100, random_state=42)
    else:
        X_sample = X
    
    # Convert to numpy array to avoid any dataframe-related issues
    X_sample_array = X_sample.values if hasattr(X_sample, 'values') else X_sample
    
    if model_type == "tree":
        # For XGBoost, use the booster directly to avoid sklearn wrapper issues
        if output_prefix == "xgboost":
            try:
                import xgboost as xgb
                # Try to get the booster object if it's a sklearn wrapper
                if hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    # Use booster directly with SHAP
                    explainer = shap.TreeExplainer(booster)
                else:
                    explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample_array)
            except (ValueError, TypeError, AttributeError) as e:
                error_msg = str(e)
                if "could not convert string to float" in error_msg or "base_score" in error_msg:
                    print(f"  ⚠ SHAP TreeExplainer incompatible with this XGBoost model")
                    print(f"  Using LIME as fallback for feature importance...")
                    
                    # Use LIME for XGBoost
                    lime_explainer = LimeTabularExplainer(
                        X_sample_array,
                        feature_names=feature_names,
                        mode='regression',
                        random_state=42
                    )
                    
                    # Get LIME explanations for all samples
                    lime_importances = np.zeros((len(X_sample_array), len(feature_names)))
                    
                    print(f"  Generating LIME explanations for {len(X_sample_array)} samples...")
                    for i in range(len(X_sample_array)):
                        exp = lime_explainer.explain_instance(
                            X_sample_array[i],
                            model.predict,
                            num_features=len(feature_names)
                        )
                        # Extract feature importances from LIME explanation
                        for feat_idx, importance in exp.as_list():
                            # feat_idx from LIME is in format "feature_name <= value"
                            # Extract just the feature name
                            feat_name = feat_idx.split('<=')[0].split('>')[0].strip()
                            if feat_name in feature_names:
                                feat_position = feature_names.index(feat_name)
                                lime_importances[i, feat_position] = importance
                        
                        if (i + 1) % 25 == 0:
                            print(f"    Processed {i + 1}/{len(X_sample_array)} samples...")
                    
                    # Use LIME importances as SHAP values for consistency
                    shap_values = lime_importances
                    print(f"  ✓ LIME analysis complete")
                else:
                    raise
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_array)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample_array, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{output_prefix}_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved summary plot to {OUTPUT_DIR / f'{output_prefix}_summary.png'}")
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample_array, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{output_prefix}_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved bar plot to {OUTPUT_DIR / f'{output_prefix}_bar.png'}")
    
    # Calculate feature importance
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # For multi-output models
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    importance_path = OUTPUT_DIR / f"{output_prefix}_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Saved feature importance to {importance_path}")
    
    return feature_importance


def main():
    """Main function to generate SHAP analysis for all models."""
    print("=" * 60)
    print("SHAP Feature Importance Analysis")
    print("=" * 60)
    
    # Load processed data
    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    processed_file = processed_dir / "processed_aqi_rawalpindi_20250101-20260110.csv"
    
    if not processed_file.exists():
        print(f"Error: Processed data not found at {processed_file}")
        return
    
    df = pd.read_csv(processed_file)
    
    # Get feature columns
    exclude_cols = ['AQI', 'AQI_Category', 'timestamp', 'city', 'latitude', 'longitude']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    
    # Analyze each model
    models = ["random_forest", "lightgbm", "xgboost"]
    
    for model_name in models:
        try:
            print(f"\n{'=' * 60}")
            print(f"Analyzing {model_name}...")
            print(f"{'=' * 60}")
            
            model, model_type, feature_cols_ordered = load_model_and_features(model_name)
            
            # Ensure feature order matches
            X_ordered = X[feature_cols_ordered]
            
            feature_importance = generate_shap_analysis(
                X_ordered, model, model_type, feature_cols_ordered,
                output_prefix=model_name
            )
            
            print(f"\nTop 10 Most Important Features for {model_name}:")
            print(feature_importance.head(10).to_string(index=False))
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 60}")
    print("SHAP analysis complete!")
    print(f"Reports saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()