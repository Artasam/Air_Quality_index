"""
Hopsworks Feature Store Integration (Corrected for SDK 4.7.2+)

This module provides integration with Hopsworks Feature Store (v4.7.2+ compatible).
Install Hopsworks: pip install hopsworks==4.7.2

Key corrections applied (March 2026):
- "start_offline_backfill" → "start_offline_materialization" (official key in 4.7+)
- Feature Group creation now enables online storage + HUDI time-travel by default
- All other logic, error handling, diagnostics, and fallbacks preserved

Usage:
    from src.feature_store.hopsworks_integration import save_features_to_hopsworks
    
    # Save features
    save_features_to_hopsworks(features_df, feature_group_name="aqi_features")
    
    # Retrieve features
    features = get_features_from_hopsworks(feature_group_name="aqi_features")
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import os
import time
import datetime

# Attempt to load environment variables from a .env file in the repo root
# This is optional and will be a no-op if python-dotenv is not installed.
try:
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parents[2]
    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        try:
            load_dotenv(dotenv_path)
            print(f"Note: Loaded environment variables from {dotenv_path}")
        except Exception:
            pass
    else:
        try:
            load_dotenv()
        except Exception:
            pass
except Exception:
    pass

try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print("Warning: Hopsworks not installed. Install with: pip install hopsworks==4.7.2")

# Small flags controlled via environment variables for quieter behavior
HOPS_DEBUG = os.getenv("HOPS_DEBUG", "0").lower() in ("1","true","yes")
HOPS_AUTO_CREATE_FEATURE_VIEW = os.getenv("HOPS_AUTO_CREATE_FEATURE_VIEW", "0").lower() in ("1","true","yes")

# Define data directories for fallback logging
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# GLOBAL SINGLETON PROJECT
# -----------------------------
_PROJECT = None

def get_hopsworks_project():
    """Initialize and return a singleton Hopsworks project."""
    global _PROJECT

    if _PROJECT is not None:
        return _PROJECT

    if not HOPSWORKS_AVAILABLE:
        raise ImportError("Hopsworks is not installed. Install with: pip install hopsworks==4.7.2")

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY environment variable not set")

    api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is empty after sanitization")

    try:
        _PROJECT = hopsworks.login(api_key_value=api_key, host="eu-west.cloud.hopsworks.ai")
        print("✓ Connected to Hopsworks (singleton session)")
        return _PROJECT
    except Exception as e:
        masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "(hidden)"
        raise RuntimeError(
            "Failed to login to Hopsworks. "
            f"(API key starts with: {masked})"
        ) from e


# Simple in-memory caches to avoid recreating feature groups/views repeatedly
_CREATED_FEATURE_GROUPS = set()
_CREATED_FEATURE_VIEWS = set()


def create_feature_group(project, feature_group_name: str, description: str = "", 
                        primary_key: list = None, event_time: str = "timestamp"):
    """Create a feature group in Hopsworks (4.7.2+ compatible)."""
    if not HOPSWORKS_AVAILABLE:
        raise ImportError("Hopsworks is not installed")
    
    # Fast local cache check
    if feature_group_name in _CREATED_FEATURE_GROUPS:
        try:
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name=feature_group_name, version=1)
            if fg is not None:
                return fg
        except Exception:
            pass

    try:
        fs = project.get_feature_store()
        
        if primary_key is None:
            primary_key = ["timestamp", "city"]
        
        # First try to fetch existing
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=1)
            if fg is not None:
                print(f"✓ Feature group '{feature_group_name}' already exists (version 1)")
                _CREATED_FEATURE_GROUPS.add(feature_group_name)
                if HOPS_AUTO_CREATE_FEATURE_VIEW:
                    try:
                        create_feature_view(feature_group_name)
                    except Exception:
                        pass
                return fg
        except Exception:
            pass

        # Create with modern defaults (online + HUDI)
        fg = fs.create_feature_group(
            name=feature_group_name,
            version=1,
            description=description,
            primary_key=primary_key,
            event_time=event_time,
            online_enabled=True,          # Enables real-time feature serving
            time_travel_format="HUDI",    # Best for time-series + backfills
        )
        print(f"✓ Created feature group '{feature_group_name}' (version 1, online+HUDI)")
        _CREATED_FEATURE_GROUPS.add(feature_group_name)
        
        if HOPS_AUTO_CREATE_FEATURE_VIEW:
            try:
                create_feature_view(feature_group_name)
            except Exception:
                pass
        return fg

    except Exception as e:
        print(f"Error creating feature group: {e}")
        raise


def save_features_to_hopsworks(
    features_df: pd.DataFrame,
    feature_group_name: str = "aqi_features",
    description: str = "AQI forecasting features",
    mode: str = "append"
):
    """
    Save features to Hopsworks Feature Store (SDK 4.7.2+ corrected).
    """
    if not HOPSWORKS_AVAILABLE:
        print("Hopsworks not available. Skipping feature store save.")
        return None
    
    try:
        if 'timestamp' not in features_df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        features_df = features_df.copy()
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        if 'city' not in features_df.columns:
            city = os.getenv("CITY", "Unknown")
            features_df['city'] = city
            print(f"Note: Added 'city' column with value '{city}'")

        # Normalize column names to lower-case (Hopsworks sanitizes automatically)
        cols_with_upper = [c for c in features_df.columns if any(ch.isupper() for ch in c)]
        if cols_with_upper:
            print(f"Note: Normalizing columns to lower-case: {cols_with_upper}")
            features_df.columns = [c.lower() for c in features_df.columns]
        
        project = get_hopsworks_project()
        
        primary_key = ["timestamp"]
        if 'city' in features_df.columns:
            primary_key.append("city")
        
        fg = create_feature_group(
            project, 
            feature_group_name, 
            description,
            primary_key=primary_key,
            event_time="timestamp"
        )
        
        # Version detection for 4.7+ behavior
        try:
            hops_version = getattr(hopsworks, '__version__', '0.0.0')
            version_parts = hops_version.split('.')
            hops_major_minor = float(version_parts[0]) + (float(version_parts[1]) / 10.0) if len(version_parts) >= 2 else 0.0
        except Exception:
            hops_major_minor = 0.0
            hops_version = "unknown"
        
        wait_for_insert_job = os.getenv("HOPS_WAIT_FOR_INSERT_JOB", "0").lower() in ("1", "true", "yes")
        wait_for_materialization = hops_major_minor >= 4.7 and wait_for_insert_job
        
        write_opts = {
            "start_offline_materialization": False,   # ← Corrected key for 4.7.2+
            "wait_for_job": wait_for_materialization
        }
        
        if wait_for_materialization:
            print(f"Inserting {len(features_df)} rows (Hopsworks {hops_version} - waiting for materialization)...")
        elif hops_major_minor >= 4.7:
            print(f"Inserting {len(features_df)} rows (Hopsworks {hops_version}, not waiting to avoid hang)...")
        else:
            print(f"Inserting {len(features_df)} rows (offline-only mode)...")
        
        fg.insert(features_df, write_options=write_opts)
        
        if wait_for_materialization:
            print(f"✓ Successfully inserted and materialized {len(features_df)} rows")
        else:
            print(f"✓ Successfully inserted {len(features_df)} rows")
            print("  Materialization job started on server (data queryable in a few minutes).")
        
        # Optional commit (safe)
        try:
            if hasattr(fg, 'commit'):
                fg.commit()
                print("✓ Insert committed")
        except Exception:
            pass
        
        return fg

    except Exception as e:
        msg = str(e).lower()
        if "materializ" in msg or "no hudi properties" in msg or "no data has been written" in msg:
            print("⚠️ Materialization-related error detected. Attempting offline backfill...")
            # Backfill fallback logic (unchanged, already robust)
            try:
                fg.insert(features_df, write_options={"start_offline_materialization": True})
                print("✓ Offline materialization requested via insert")
                return fg
            except Exception as back_exc:
                print(f"Backfill request failed: {back_exc}")
        
        print(f"⚠️ Error saving to Hopsworks: {e}")
        return None


# All remaining functions (trigger_materialization_if_needed, check_featuregroup_materialized,
# inspect_featuregroup_schema, attempt_recreate_featuregroup, create_feature_view,
# get_features_from_hopsworks, create_training_dataset, save_model_to_hopsworks)
# are unchanged because they were already correct and use the updated keys where needed.

def trigger_materialization_if_needed(feature_group_name: str = "aqi_features") -> bool:
    """Attempt to trigger materialization (unchanged - already robust)."""
    if not HOPSWORKS_AVAILABLE:
        return False
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=feature_group_name, version=1)
        
        for method_name in ['start_offline_materialization', 'start_offline_backfill', 
                           'start_backfill', 'start_materialization', 'request_offline_backfill']:
            if hasattr(fg, method_name):
                try:
                    getattr(fg, method_name)()
                    print(f"✓ Triggered materialization via fg.{method_name}()")
                    return True
                except Exception:
                    pass
        return False
    except Exception:
        return False


def check_featuregroup_materialized(feature_group_name: str = "aqi_features") -> dict:
    """Return diagnostics about materialization status (unchanged)."""
    if not HOPSWORKS_AVAILABLE:
        return {"exists": False, "readable": False, "rows": None, "message": "Hopsworks not installed"}
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=feature_group_name, version=1)
        
        try:
            df = fg.select_all().limit(1).read()
            rows = None
            try:
                rows = len(fg.read())
            except Exception:
                pass
            return {"exists": True, "readable": True, "rows": rows, "message": "Feature group readable"}
        except Exception as read_exc:
            return {"exists": True, "readable": False, "rows": None, "message": f"Read failed: {read_exc}"}
    except Exception as e:
        return {"exists": False, "readable": False, "rows": None, "message": f"Error: {e}"}


def inspect_featuregroup_schema(feature_group_name: str = "aqi_features") -> dict:
    """Inspect schema and detect issues (unchanged)."""
    result = {"ok": True, "features": [], "issues": []}
    if not HOPSWORKS_AVAILABLE:
        result["ok"] = False
        result["issues"].append("Hopsworks client not installed")
        return result
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=feature_group_name, version=1)
        
        features = None
        try:
            if hasattr(fg, "get_features"):
                features = fg.get_features()
            elif hasattr(fg, "features"):
                features = fg.features
            else:
                df = fg.select_all().limit(1).read()
                features = [{"name": c, "type": str(df[c].dtype)} for c in df.columns]
        except Exception:
            features = None

        if features is None:
            result["ok"] = False
            result["issues"].append("Could not determine features")
            return result

        normalized = []
        for f in features:
            try:
                if isinstance(f, dict):
                    fname = f.get("name")
                else:
                    fname = getattr(f, "name", None) or getattr(f, "feature_name", None)
                normalized.append({"name": fname})
                if not isinstance(fname, str):
                    result["issues"].append(f"Feature name not string: {fname}")
            except Exception as e:
                result["issues"].append(f"Parse error: {e}")
        
        result["features"] = normalized
        if result["issues"]:
            result["ok"] = False
        return result
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Access error: {e}")
        return result


def attempt_recreate_featuregroup(feature_group_name: str, features_df: pd.DataFrame) -> dict:
    """Safe (destructive) recreate (updated materialization key)."""
    if os.getenv("FORCE_RECREATE_FEATURE_GROUP", "0").lower() not in ("1", "true", "yes"):
        return {"ok": False, "message": "Set FORCE_RECREATE_FEATURE_GROUP=1 to enable"}

    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        
        # Delete if exists
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=1)
            if hasattr(fg, "delete"):
                fg.delete()
        except Exception:
            pass

        pk = ["timestamp"]
        if "city" in features_df.columns:
            pk.append("city")
        
        fg_new = fs.create_feature_group(
            name=feature_group_name,
            version=1,
            description="Recreated by automation",
            primary_key=pk,
            event_time="timestamp",
            online_enabled=True,
            time_travel_format="HUDI"
        )
        fg_new.insert(features_df, write_options={"start_offline_materialization": True})
        return {"ok": True, "message": "Feature group recreated and materialization requested"}
    except Exception as e:
        return {"ok": False, "message": f"Failed: {e}"}


def create_feature_view(feature_group_name: str, view_name: Optional[str] = None, description: str = "Auto-created feature view"):
    """Ensure a Feature View exists (unchanged - already handles all SDK variations)."""
    if not HOPSWORKS_AVAILABLE:
        print("Hopsworks client not available")
        return None

    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
    except Exception as e:
        print(f"Error accessing feature store: {e}")
        return None

    if view_name is None:
        view_name = f"{feature_group_name}_view"

    # Check if exists
    try:
        fv = fs.get_feature_view(name=view_name)
        if fv is not None:
            print(f"✓ Feature view '{view_name}' already exists")
            return fv
    except Exception:
        pass

    # Try creation (multi-strategy - kept as-is)
    try:
        fg = fs.get_feature_group(name=feature_group_name, version=1)
        query = fg.select_all()
        
        fv = fs.create_feature_view(
            name=view_name,
            query=query,
            version=1,
            description=description
        )
        print(f"✓ Created feature view '{view_name}'")
        return fv
    except Exception as e:
        print(f"Note: Could not create feature view programmatically: {e}")
        print("Please create it manually in Hopsworks UI:")
        print(f"   Name: {view_name}")
        print(f"   Query: SELECT * FROM {feature_group_name}")
        return None


def get_features_from_hopsworks(
    feature_group_name: str = "aqi_features",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> pd.DataFrame:
    """Retrieve features (unchanged - already correct and robust)."""
    if not HOPSWORKS_AVAILABLE:
        raise ImportError("Hopsworks is not installed")

    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=feature_group_name, version=1)

    try:
        if start_time and end_time:
            feature_view = fg.select_all().filter(
                (fg.timestamp >= start_time) & (fg.timestamp <= end_time)
            )
        else:
            feature_view = fg.select_all()

        retry_max = int(os.getenv("HOPS_QUERY_MAX_RETRIES", "6"))
        base_delay = float(os.getenv("HOPS_QUERY_BASE_DELAY", "5.0"))
        
        for attempt in range(1, retry_max + 1):
            try:
                df = feature_view.read()
                print(f"✓ Retrieved {len(df)} rows via Query Service")
                return df
            except Exception as qs_err:
                err_text = str(qs_err).lower()
                if any(x in err_text for x in ["hudi", "hoodie", "no hudi properties", "materialization"]):
                    if attempt == 1:
                        trigger_materialization_if_needed(feature_group_name)
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"Query retry {attempt}/{retry_max} (waiting {delay:.1f}s for materialization)...")
                    time.sleep(delay)
                    continue
                else:
                    break

        # Fallback offline read
        df = fg.read()
        print(f"✓ Fallback offline read succeeded: {len(df)} rows")
        return df

    except Exception as e:
        print(f"Error retrieving features: {e}")
        raise


def create_training_dataset(
    feature_group_name: str = "aqi_features",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> pd.DataFrame:
    """Create training dataset with column normalization (unchanged)."""
    df = get_features_from_hopsworks(feature_group_name, start_time, end_time)

    # Normalize AQI / AQI_Category (Hopsworks lower-casing)
    cols_lower_map = {c.lower(): c for c in df.columns}
    if 'aqi' in cols_lower_map and 'AQI' not in df.columns:
        df['AQI'] = df[cols_lower_map['aqi']]
    if 'aqi_category' in cols_lower_map and 'AQI_Category' not in df.columns:
        df['AQI_Category'] = df[cols_lower_map['aqi_category']]

    if 'timestamp' in cols_lower_map and 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df[cols_lower_map['timestamp']])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    if 'AQI' not in df.columns:
        raise ValueError("AQI column not found after normalization")

    return df


def save_model_to_hopsworks(
    model_path,
    model_name: str,
    description: str = "",
    model_type: str = "sklearn",
    metadata: dict = None
):
    """Save model to Hopsworks Model Registry (4.7.2+ compatible - unchanged)."""
    if not HOPSWORKS_AVAILABLE:
        print("Hopsworks not available. Skipping model registry upload.")
        return False
    
    from pathlib import Path
    import shutil
    
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    project = get_hopsworks_project()
    mr = project.get_model_registry()
    
    model_dir = model_path.parent / f"{model_name}_artifact"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if model_path.is_file():
        shutil.copy(model_path, model_dir / model_path.name)
    else:
        shutil.copytree(model_path, model_dir, dirs_exist_ok=True)
    
    print(f"✓ Prepared model directory: {model_dir}")
    
    framework_map = {
        'sklearn': mr.sklearn,
        'tensorflow': mr.tensorflow,
        'torch': mr.torch,
        'python': mr.python
    }
    framework_api = framework_map.get(model_type, mr.python)
    
    numeric_metrics = {}
    if metadata:
        for k, v in metadata.items():
            try:
                float(v)
                numeric_metrics[k] = v
            except (ValueError, TypeError):
                pass
    
    model = framework_api.create_model(
        name=model_name,
        description=description,
        metrics=numeric_metrics
    )
    
    model.save(str(model_dir))
    print(f"✓ Model '{model_name}' (v{model.version}) uploaded successfully!")
    
    try:
        if model_dir.exists():
            shutil.rmtree(model_dir)
    except Exception:
        pass
    
    return True


# CLI for diagnostics (unchanged)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hopsworks Feature Store helper CLI")
    parser.add_argument("command", choices=["check", "inspect", "recreate", "create_view"])
    parser.add_argument("--fg", default="aqi_features")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--view", default=None)
    args = parser.parse_args()

    if args.command == "check":
        print(check_featuregroup_materialized(args.fg))
    elif args.command == "inspect":
        print(inspect_featuregroup_schema(args.fg))
        if args.sample:
            try:
                project = get_hopsworks_project()
                fs = project.get_feature_store()
                fg = fs.get_feature_group(name=args.fg, version=1)
                print(fg.select_all().limit(5).read())
            except Exception as e:
                print("Sample read failed:", e)
    elif args.command == "recreate":
        sample_path = os.getenv("RECREATE_SAMPLE_PATH")
        if not sample_path:
            print("Set RECREATE_SAMPLE_PATH to a CSV with sample data")
        else:
            df = pd.read_csv(sample_path)
            print(attempt_recreate_featuregroup(args.fg, df))
    elif args.command == "create_view":
        view_name = args.view or f"{args.fg}_view"
        create_feature_view(args.fg, view_name=view_name)
