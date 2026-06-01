"""
Hopsworks Feature Store Integration (Corrected for SDK 4.7.2+)

This module provides integration with Hopsworks Feature Store (v4.7.2+ compatible).
Install Hopsworks: pip install hopsworks==4.7.2

Key corrections applied (March 2026):
- "start_offline_backfill" → "start_offline_materialization" (official key in 4.7+)
- Feature Group creation now enables online storage + HUDI time-travel by default
- FIXED (June 2026): start_offline_materialization set to FALSE on hourly inserts.
  Spawning a Spark materialization job every hour caused "materialization already running"
  conflicts. Online store is updated immediately on insert. Offline materialization is
  now triggered ONCE per day by the training pipeline before it reads training data.
- All other logic, error handling, diagnostics, and fallbacks preserved

Usage:
    from src.feature_store.hopsworks_integration import save_features_to_hopsworks

    # Save features (hourly — online store only, no Spark job)
    save_features_to_hopsworks(features_df, feature_group_name="aqi_features")

    # Retrieve features for training (call trigger_offline_materialization_for_training first)
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
HOPS_DEBUG = os.getenv("HOPS_DEBUG", "0").lower() in ("1", "true", "yes")
HOPS_AUTO_CREATE_FEATURE_VIEW = os.getenv("HOPS_AUTO_CREATE_FEATURE_VIEW", "0").lower() in ("1", "true", "yes")

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

    IMPORTANT: This function does NOT trigger offline materialization.
    The online store is updated immediately on every insert (fast, no Spark job).
    Offline materialization is triggered once per day by the training pipeline.
    This prevents the "materialization already running" error on hourly runs.
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

        # ---------------------------------------------------------------
        # FIX: Do NOT start offline materialization on every hourly insert.
        #
        # Root cause of the "materialization already running" error:
        #   - start_offline_materialization: True spawns a new Spark job
        #     on EVERY insert call.
        #   - The hourly feature pipeline runs every 60 minutes, but each
        #     Spark job takes longer than that on the free Hopsworks tier.
        #   - Jobs pile up → Hopsworks raises "materialization already running"
        #     on the next insert → pipeline crashes → dashboard goes stale.
        #
        # Correct pattern (per Hopsworks docs):
        #   - Insert with start_offline_materialization: False  → online store
        #     is updated immediately, no Spark job, no conflict.
        #   - Trigger materialization ONCE per day from the training pipeline
        #     (see trigger_offline_materialization_for_training below) before
        #     reading the offline store for model training.
        # ---------------------------------------------------------------
        write_opts = {
            "start_offline_materialization": False,  # ← KEY FIX: no Spark job on insert
            "wait_for_job": False,
        }

        print(f"Inserting {len(features_df)} rows into online store "
              f"(offline materialization intentionally skipped to prevent job conflicts)...")

        fg.insert(features_df, write_options=write_opts)

        print(f"✓ Inserted {len(features_df)} rows. Online store updated immediately.")
        print("  Offline materialization will be triggered once by the daily training pipeline.")

        # Optional commit (safe no-op if unsupported)
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
            print("⚠️ Materialization-related error detected. Retrying with materialization disabled...")
            try:
                # ── FIX: also use False in the fallback, not True ──
                fg.insert(features_df, write_options={
                    "start_offline_materialization": False,
                    "wait_for_job": False,
                })
                print("✓ Fallback insert succeeded (offline materialization skipped)")
                return fg
            except Exception as back_exc:
                print(f"Fallback insert failed: {back_exc}")

        print(f"⚠️ Error saving to Hopsworks: {e}")
        return None


def trigger_offline_materialization_for_training(
    feature_group_name: str = "aqi_features",
    wait: bool = True
) -> bool:
    """
    Trigger offline materialization ONCE before reading training data.

    Call this from training_pipeline.py at the START of each daily training run,
    BEFORE calling get_features_from_hopsworks() or create_training_dataset().

    DO NOT call this from the hourly feature pipeline — that is what caused the
    "materialization already running" error in the first place.

    Pattern:
        # In training_pipeline.py:
        from src.feature_store.hopsworks_integration import (
            trigger_offline_materialization_for_training,
            create_training_dataset,
        )
        trigger_offline_materialization_for_training()   # run ONCE
        df = create_training_dataset()                   # then read
    """
    if not HOPSWORKS_AVAILABLE:
        print("Hopsworks not available. Skipping materialization trigger.")
        return False

    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=feature_group_name, version=1)

        print(f"Triggering offline materialization for '{feature_group_name}' "
              f"({'blocking until complete' if wait else 'non-blocking'})...")

        # Use start_offline_materialization method if available (preferred, SDK 4.x)
        for method_name in ['start_offline_materialization', 'start_offline_backfill',
                            'start_backfill', 'start_materialization']:
            if hasattr(fg, method_name):
                try:
                    job = getattr(fg, method_name)()
                    if wait and job is not None and hasattr(job, 'run'):
                        execution = job.run(await_termination=True)
                        success = getattr(execution, 'success', True)
                        if success:
                            print(f"✓ Offline materialization complete (via fg.{method_name}())")
                        else:
                            print(f"⚠️ Materialization job finished but reported failure. "
                                  f"Check Hopsworks UI for details.")
                    else:
                        print(f"✓ Offline materialization triggered (via fg.{method_name}()). "
                              f"Running in background.")
                    return True
                except Exception as method_err:
                    print(f"  fg.{method_name}() failed: {method_err}. Trying next method...")
                    continue

        # Fallback: insert an empty frame just to get the job handle
        print("Direct materialization method not available. Using insert-based trigger...")
        job, _ = fg.insert(
            pd.DataFrame(),
            write_options={
                "start_offline_materialization": True,
                "wait_for_job": wait,
            }
        )
        print("✓ Offline materialization triggered via insert fallback.")
        return True

    except Exception as e:
        print(f"⚠️ Could not trigger offline materialization: {e}")
        print("  Training will attempt a direct read anyway — data may be slightly stale.")
        return False


def trigger_materialization_if_needed(feature_group_name: str = "aqi_features") -> bool:
    """Attempt to trigger materialization (kept for backward compatibility)."""
    return trigger_offline_materialization_for_training(
        feature_group_name=feature_group_name,
        wait=False
    )


def check_featuregroup_materialized(feature_group_name: str = "aqi_features") -> dict:
    """Return diagnostics about materialization status."""
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
    """Inspect schema and detect issues."""
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
    """Safe (destructive) recreate — requires FORCE_RECREATE_FEATURE_GROUP=1."""
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
        # Initial population triggers one materialization (acceptable for a one-off recreate)
        fg_new.insert(features_df, write_options={
            "start_offline_materialization": True,
            "wait_for_job": True,
        })
        return {"ok": True, "message": "Feature group recreated and initial materialization complete"}
    except Exception as e:
        return {"ok": False, "message": f"Failed: {e}"}


def create_feature_view(feature_group_name: str, view_name: Optional[str] = None,
                        description: str = "Auto-created feature view"):
    """Ensure a Feature View exists."""
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
    """
    Retrieve features from the offline store for training.

    Important: Call trigger_offline_materialization_for_training() before this
    to ensure the offline store reflects the latest inserts.
    """
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
                        print("Offline store not ready — triggering materialization and retrying...")
                        trigger_offline_materialization_for_training(feature_group_name, wait=True)
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"Query retry {attempt}/{retry_max} (waiting {delay:.1f}s)...")
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
    """
    Create training dataset with column normalization.

    Triggers offline materialization automatically before reading.
    """
    print("Ensuring offline store is up to date before reading training data...")
    trigger_offline_materialization_for_training(feature_group_name, wait=True)

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
    """Save model to Hopsworks Model Registry (4.7.2+ compatible)."""
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


# CLI for diagnostics
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hopsworks Feature Store helper CLI")
    parser.add_argument("command", choices=["check", "inspect", "recreate", "create_view", "materialize"])
    parser.add_argument("--fg", default="aqi_features")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--view", default=None)
    parser.add_argument("--no-wait", action="store_true",
                        help="For 'materialize': trigger without waiting for completion")
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
    elif args.command == "materialize":
        success = trigger_offline_materialization_for_training(
            feature_group_name=args.fg,
            wait=not args.no_wait
        )
        print("✓ Done" if success else "⚠️ Failed — check logs above")
