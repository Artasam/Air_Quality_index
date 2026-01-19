"""
Hopsworks Feature Store Integration

This module provides integration with Hopsworks Feature Store.
Install Hopsworks: pip install hopsworks

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
            # If loading fails for any reason, continue without failing hard
            pass
    else:
        # Fallback to default load_dotenv() behavior which searches current working dir and parent directories
        try:
            load_dotenv()
        except Exception:
            pass
except Exception:
    # python-dotenv not installed; continue silently
    pass

try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print("Warning: Hopsworks not installed. Install with: pip install hopsworks")

# Small flags controlled via environment variables for quieter behavior
HOPS_DEBUG = os.getenv("HOPS_DEBUG", "0").lower() in ("1","true","yes")
HOPS_AUTO_CREATE_FEATURE_VIEW = os.getenv("HOPS_AUTO_CREATE_FEATURE_VIEW", "0").lower() in ("1","true","yes")

# Define data directories for fallback logging (same as in features_extraction.py)
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



def get_hopsworks_project():
    """Initialize and return Hopsworks project."""
    if not HOPSWORKS_AVAILABLE:
        raise ImportError("Hopsworks is not installed. Install with: pip install hopsworks")
    
    # Get API key from environment and sanitize (strip whitespace and quotes)
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY environment variable not set")
    api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY environment variable is empty after sanitization. Ensure it is set correctly in the environment or .env file.")

    try:
        project = hopsworks.login(api_key_value=api_key)
        return project
    except Exception as e:
        # Provide actionable guidance without printing secrets
        masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "(key hidden)"
        raise RuntimeError(
            "Failed to login to Hopsworks. Check your API key and project access. "
            f"(Provided API key starts with: {masked})"
        ) from e


# Simple in-memory caches to avoid recreating feature groups/views repeatedly during a single run
_CREATED_FEATURE_GROUPS = set()
_CREATED_FEATURE_VIEWS = set()


def create_feature_group(project, feature_group_name: str, description: str = "", 
                        primary_key: list = None, event_time: str = "timestamp"):
    """Create a feature group in Hopsworks.

    This function is idempotent within a single Python process: it keeps a
    lightweight cache of feature groups that have already been confirmed/created
    so repeated calls during the same run will not attempt duplicate creation
    calls to the remote API.
    """
    if not HOPSWORKS_AVAILABLE:
        raise ImportError("Hopsworks is not installed")
    
    # Fast local cache check to avoid duplicate network calls
    if feature_group_name in _CREATED_FEATURE_GROUPS:
        try:
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name=feature_group_name, version=1)
            if fg is not None:
                return fg
        except Exception:
            # If cache was stale (e.g., failed to fetch FG), continue to normal flow
            pass

    try:
        # Get or create feature store
        fs = project.get_feature_store()
        
        # Default primary key
        if primary_key is None:
            primary_key = ["timestamp", "city"]
        
        # First try to fetch existing feature group (preferred)
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=1)
            if fg is not None:
                print(f"✓ Feature group '{feature_group_name}' already exists (version 1)")
                _CREATED_FEATURE_GROUPS.add(feature_group_name)
                # Optionally ensure feature view exists only if env enables it
                if os.getenv("HOPS_AUTO_CREATE_FEATURE_VIEW", "0").lower() in ("1", "true", "yes"):
                    try:
                        create_feature_view(feature_group_name)
                    except Exception:
                        pass
                return fg
        except Exception:
            # Not fatal - we'll attempt to create the feature group below
            pass

        # Attempt to create the feature group
        try:
            fg = fs.create_feature_group(
                name=feature_group_name,
                version=1,
                description=description,
                primary_key=primary_key,
                event_time=event_time
            )
            print(f"✓ Created feature group '{feature_group_name}' (version 1)")
            _CREATED_FEATURE_GROUPS.add(feature_group_name)
            # Optionally ensure feature view exists only if env enables it
            if os.getenv("HOPS_AUTO_CREATE_FEATURE_VIEW", "0").lower() in ("1", "true", "yes"):
                try:
                    create_feature_view(feature_group_name)
                except Exception:
                    pass
            return fg
        except Exception as create_error:
            # If creation fails, try to get existing group with different version
            print(f"Note: {create_error}")
            try:
                fg_alt = fs.get_feature_group(name=feature_group_name)
                if fg_alt is not None:
                    print(f"✓ Found existing feature group '{feature_group_name}' (alternative lookup)")
                    _CREATED_FEATURE_GROUPS.add(feature_group_name)
                    return fg_alt
            except Exception:
                pass
            raise

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
    Save features to Hopsworks Feature Store.
    
    Args:
        features_df: DataFrame with features (must include 'timestamp' and 'city' columns)
        feature_group_name: Name of the feature group
        description: Description of the feature group
        mode: Insert mode - 'append' (default) or 'overwrite'
    """
    if not HOPSWORKS_AVAILABLE:
        print("Hopsworks not available. Skipping feature store save.")
        return None
    
    try:
        # Validate required columns
        if 'timestamp' not in features_df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        # Work on a copy
        features_df = features_df.copy()
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # Ensure city column exists
        if 'city' not in features_df.columns:
            # Try to get from environment or use default
            import os
            city = os.getenv("CITY", "Unknown")
            features_df['city'] = city
            print(f"Note: Added 'city' column with value '{city}'")

        # Normalize column names to lower-case to avoid FeatureStore warnings
        # Hopsworks sanitizes column names to lower case; proactively rename to avoid warnings.
        cols_with_upper = [c for c in features_df.columns if any(ch.isupper() for ch in c)]
        if cols_with_upper:
            print(f"Note: Normalizing columns to lower-case before saving to Hopsworks: {cols_with_upper}")
            features_df.columns = [c.lower() for c in features_df.columns]
        
        project = get_hopsworks_project()
        
        # Determine primary key based on available columns
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
        
        # DEBUG: Inspect feature group object before attempting insert (only when HOPS_DEBUG=1)
        try:
            if HOPS_DEBUG:
                print(f"DEBUG: fg type={type(fg)}")
                try:
                    print(f"DEBUG: fg repr: {repr(fg)[:200]}")
                except Exception:
                    pass
                print(f"DEBUG: has insert: {hasattr(fg, 'insert')}")
                print(f"DEBUG: getattr insert: {getattr(fg, 'insert', None)}")
            # Insert features with offline-only mode (bypasses Kafka entirely)
            try:
                # For CI/CD pipelines: Use OFFLINE-ONLY insertion (no Kafka dependency)
                # This writes directly to Hudi/HDFS files without requiring Kafka brokers
                
                # The key insight: set `write_options` with ONLY offline backfill, 
                # and the HSFS SDK will skip Kafka entirely and write directly to Hudi
                write_opts = {
                    "start_offline_backfill": True,
                    "wait_for_job": False  # Don't block waiting for Hudi job; fire and forget
                }
                
                print(f"Inserting {len(features_df)} rows using offline-only mode (no Kafka dependency)...")
                fg.insert(features_df, write_options=write_opts)
                print(f"✓ Successfully inserted {len(features_df)} rows to Hopsworks feature group '{feature_group_name}'")
                print("  Data is being written to Hudi/HDFS storage asynchronously.")
                print("  Data will be queryable in 2-5 minutes (no Kafka required).")

                # Attempt to commit if supported
                try:
                    if hasattr(fg, 'commit'):
                        fg.commit()
                        print("✓ Insert committed successfully")
                except Exception as commit_exc:
                    print(f"Note: commit() not available or failed: {commit_exc}")

                # Return immediately (don't wait for materialization—let Hopsworks handle it asynchronously)
                return fg

            except Exception as insert_exc:
                # Insert failed - re-raise the error
                err_type = type(insert_exc).__name__
                err_msg = str(insert_exc)
                print(f"⚠️  Insert failed ({err_type}): {err_msg}")
                raise insert_exc
        except AttributeError as ae:
            # Print detailed diagnostics and re-raise
            import traceback
            print("⚠️  AttributeError during fg.insert:")
            traceback.print_exc()
            print(f"DEBUG: dir(fg) (partial): {[name for name in dir(fg) if 'insert' in name or 'write' in name][:50]}")
            raise
        except ModuleNotFoundError as mnfe:
            # Common missing dependency is confluent-kafka required by HSFS for ingestion
            msg = str(mnfe)
            if 'Confluent Kafka' in msg or 'confluent-kafka' in msg.lower():
                print("⚠️  Missing dependency: confluent-kafka is required for Hopsworks feature ingestion.")
                print("   Install with: pip install confluent-kafka")
                print("   Or install the Hopsworks python extras: pip install 'hopsworks[python]'")
            else:
                print(f"⚠️  Module not found: {mnfe}")
            print(f"   Error type: {type(mnfe).__name__}")
            return None

    except ValueError as ve:
        print(f"⚠️  Validation error: {ve}")
        return None
    except Exception as e:
        msg = str(e)
        # Detect materialization-related problems robustly (handles typos like 'materiaalization')
        if "materializ" in msg.lower() or "no materialization" in msg.lower() or "no materialization job" in msg.lower():
            print("⚠️  Detected materialization-related error from Hopsworks:", msg)
            print("   Attempting to request an offline backfill to materialize the feature group (best-effort).")

            # Try multiple backfill strategies across different HSFS SDK versions
            backfill_attempted = False
            backfill_success = False

            # 1) If we have a feature group object, try common FG methods
            try:
                if 'fg' in locals() and fg is not None:
                    for method_name in [
                        'start_offline_backfill', 'start_backfill', 'start_materialization',
                        'start_feature_group_offline_backfill', 'request_offline_backfill',
                    ]:
                        if hasattr(fg, method_name):
                            backfill_attempted = True
                            try:
                                method = getattr(fg, method_name)
                                # Try calling with and without arguments (some SDKs accept none)
                                try:
                                    method()
                                except TypeError:
                                    try:
                                        method(features_df)
                                    except TypeError:
                                        try:
                                            method(feature_group_name)
                                        except Exception as inner:
                                            raise inner
                                print(f"✓ Offline backfill requested via fg.{method_name}()")
                                backfill_success = True
                                break
                            except Exception as inner_exc:
                                if HOPS_DEBUG:
                                    print(f"fg.{method_name}() attempt failed: {inner_exc}")
                                continue

                    # As a last resort, try an insert with start_offline_backfill flag again
                    if not backfill_success:
                        try:
                            fg.insert(features_df, write_options={"start_offline_backfill": True})
                            print("✓ Offline backfill requested successfully via fg.insert(..., start_offline_backfill=True)")
                            backfill_success = True
                        except Exception as back_exc:
                            if HOPS_DEBUG:
                                print(f"fg.insert(..., start_offline_backfill=True) failed: {back_exc}")
                            else:
                                print("⚠️  Offline backfill request failed.")

            except Exception as fg_exc:
                if HOPS_DEBUG:
                    print(f"Error while attempting feature-group-level backfill methods: {fg_exc}")

            # 2) If above didn't work, try feature store level APIs
            if not backfill_success:
                try:
                    project = get_hopsworks_project()
                    fs = project.get_feature_store()
                    for method_name in [
                        'start_offline_backfill', 'start_backfill', 'request_offline_backfill', 'start_feature_group_offline_backfill'
                    ]:
                        if hasattr(fs, method_name):
                            backfill_attempted = True
                            try:
                                method = getattr(fs, method_name)
                                # Try common argument signatures
                                try:
                                    method(feature_group_name)
                                except TypeError:
                                    try:
                                        method(name=feature_group_name, version=1)
                                    except TypeError:
                                        try:
                                            method(feature_group_name, 1)
                                        except Exception as inner:
                                            raise inner
                                print(f"✓ Offline backfill requested via fs.{method_name}()")
                                backfill_success = True
                                break
                            except Exception as inner_exc:
                                if HOPS_DEBUG:
                                    print(f"fs.{method_name}() attempt failed: {inner_exc}")
                                continue
                except Exception as fs_exc:
                    if HOPS_DEBUG:
                        print(f"Error while attempting feature-store-level backfill methods: {fs_exc}")

            # Final status messaging
            if backfill_success:
                print("✓ Offline backfill requested. Please wait a few minutes and then verify materialization in the Hopsworks UI.")
                return fg if 'fg' in locals() else None
            elif backfill_attempted and not backfill_success:
                print("⚠️  Offline backfill request failed:", msg)
                print("   Suggestion: In the Hopsworks UI navigate to the feature group and run an offline backfill, or contact your Hopsworks admin.")
                return None
            else:
                print("⚠️  Could not find a supported API to request an offline backfill programmatically in this Hopsworks SDK.")
                print("   Suggestion: In the Hopsworks UI navigate to the feature group and run an offline backfill, or contact your Hopsworks admin.")
                return None

        print(f"⚠️  Error saving to Hopsworks: {e}")
        print(f"   Error type: {type(e).__name__}")
        return None


def check_featuregroup_materialized(feature_group_name: str = "aqi_features") -> dict:
    """Return simple diagnostics about a feature group's materialization status.

    Returns a dict with keys: 'exists', 'readable', 'rows' (None if unknown), 'message'.
    """
    if not HOPSWORKS_AVAILABLE:
        return {"exists": False, "readable": False, "rows": None, "message": "Hopsworks not installed"}
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=1)
        except Exception as e:
            return {"exists": False, "readable": False, "rows": None, "message": f"Feature group not found: {e}"}

        # Check readability
        try:
            df = fg.select_all().limit(1).read()
            # Try to get row count if possible
            rows = None
            try:
                # Some clients support fg.count() or reading and checking length
                rows = len(fg.read())
            except Exception:
                rows = None
            return {"exists": True, "readable": True, "rows": rows, "message": "Feature group readable via query service"}
        except Exception as read_exc:
            return {"exists": True, "readable": False, "rows": None, "message": f"Read failed: {read_exc}"}
    except Exception as e:
        return {"exists": False, "readable": False, "rows": None, "message": f"Error connecting to Hopsworks: {e}"}


# --- Additional diagnostics & safe-recreate helpers ---

def inspect_featuregroup_schema(feature_group_name: str = "aqi_features") -> dict:
    """Return the feature group's schema and detect malformed entries.

    The function inspects the feature descriptors returned by Hopsworks and
    reports any non-string feature names or unexpected structures which are a
    common cause of Query Service/DuckDB binder errors.

    Returns a dict: {"ok": bool, "features": list, "issues": list}
    """
    result = {"ok": True, "features": [], "issues": []}
    if not HOPSWORKS_AVAILABLE:
        result["ok"] = False
        result["issues"].append("Hopsworks client not installed")
        return result
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=feature_group_name, version=1)
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Could not access feature group: {e}")
        return result

    # Attempt to fetch feature descriptors via common APIs
    features = None
    try:
        if hasattr(fg, "get_features"):
            features = fg.get_features()
        elif hasattr(fg, "features"):
            features = fg.features
        else:
            # As a last resort try reading a small sample and introspect columns
            try:
                df = fg.select_all().limit(1).read()
                features = [{"name": c, "type": str(df[c].dtype)} for c in df.columns]
            except Exception:
                features = None

    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Failed to retrieve feature descriptors: {e}")
        features = None

    if features is None:
        result["ok"] = False
        result["issues"].append("Could not determine feature descriptors; feature group metadata may be corrupted.")
        return result

    # Normalize features to list of dicts
    normalized = []
    for f in features:
        try:
            if isinstance(f, dict):
                fname = f.get("name")
                ftype = f.get("type")
            else:
                # If f is a Feature object, try common attrs
                fname = getattr(f, "name", None) or getattr(f, "feature_name", None)
                ftype = getattr(f, "type", None) or getattr(f, "feature_type", None)

            normalized.append({"name": fname, "type": ftype})
            # Detect non-string feature name
            if not isinstance(fname, str):
                result["issues"].append(f"Feature name not a string: {fname} (raw: {f})")
        except Exception as e:
            result["issues"].append(f"Error parsing feature descriptor: {e} (raw: {f})")

    result["features"] = normalized
    if result["issues"]:
        result["ok"] = False
    return result


def attempt_recreate_featuregroup(feature_group_name: str, features_df: pd.DataFrame) -> dict:
    """Attempt a safe recreate of the feature group.

    This is a destructive operation (deletes and recreates the feature group).
    It will only run if the environment variable `FORCE_RECREATE_FEATURE_GROUP` is set to a truthy value.

    Returns: dict with keys {"ok": bool, "message": str}
    """
    if os.getenv("FORCE_RECREATE_FEATURE_GROUP", "0").lower() not in ("1", "true", "yes"):
        return {"ok": False, "message": "Recreate not allowed - set FORCE_RECREATE_FEATURE_GROUP=1 to enable"}

    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        # Attempt to delete feature group (different SDKs expose different APIs)
        try:
            if hasattr(fs, "delete_feature_group"):
                fs.delete_feature_group(name=feature_group_name, version=1)
            elif hasattr(fs, "drop_feature_group"):
                fs.drop_feature_group(name=feature_group_name, version=1)
            else:
                # Try on the fg object
                fg = fs.get_feature_group(name=feature_group_name, version=1)
                if hasattr(fg, "delete"):
                    fg.delete()
                else:
                    return {"ok": False, "message": "No API found to delete feature group; run manual cleanup in Hopsworks UI"}
        except Exception as del_exc:
            # Continue if deletion failed because group didn't exist or API mismatch
            print(f"Note: deletion attempt returned: {del_exc}")

        # Recreate with features_df schema
        try:
            # Determine primary key
            pk = ["timestamp"]
            if "city" in features_df.columns:
                pk.append("city")
            fg_new = fs.create_feature_group(
                name=feature_group_name,
                version=1,
                description="Recreated by automation",
                primary_key=pk,
                event_time="timestamp"
            )
            fg_new.insert(features_df, write_options={"start_offline_backfill": True})
            return {"ok": True, "message": "Feature group recreated and offline backfill requested"}
        except Exception as create_exc:
            return {"ok": False, "message": f"Failed to recreate feature group: {create_exc}"}

    except Exception as e:
        return {"ok": False, "message": f"Error connecting to Hopsworks: {e}"}


def create_feature_view(feature_group_name: str, view_name: Optional[str] = None, description: str = "Auto-created feature view"):
    """Ensure a Feature View exists for the given Feature Group.

    This function tries multiple common ways to create a Feature View across
    different Hopsworks/HSFS versions. It prints detailed diagnostic notes
    when it cannot create a view automatically.

    Returns the feature view object on success, or None if it could not be created.
    """
    if not HOPSWORKS_AVAILABLE:
        print("Hopsworks client not available; cannot create feature view.")
        return None

    import inspect

    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
    except Exception as e:
        print(f"Error accessing Hopsworks feature store for view creation: {e}")
        return None

    if view_name is None:
        view_name = f"{feature_group_name}_view"

    # 1) Check if already exists
    try:
        if hasattr(fs, "get_feature_view"):
            try:
                fv = fs.get_feature_view(name=view_name)
                if fv is not None:
                    print(f"✓ Feature view '{view_name}' already exists")
                    return fv
            except Exception:
                # Not found; we'll attempt to create one
                pass

        # 2) Try fs.create_feature_view with common argument shapes
        if hasattr(fs, "create_feature_view"):
            try:
                sig = None
                try:
                    sig = inspect.signature(fs.create_feature_view)
                    if HOPS_DEBUG:
                        print(f"DEBUG: fs.create_feature_view signature: {sig}")
                except Exception:
                    pass

                # Attempt: name + description + query
                try:
                    fv = fs.create_feature_view(name=view_name, description=description, query=f"SELECT * FROM {feature_group_name}")
                    print(f"✓ Created feature view '{view_name}' via fs.create_feature_view(name=..., query=...)")
                    return fv
                except Exception as e1:
                    msg = str(e1)
                    # Detect some common SDK incompatibility errors and give friendly guidance rather than spamming tracebacks
                    if "check_and_warn_ambiguous_features" in msg or "featurestore_id" in msg or "FeatureView.__init__" in msg:
                        print("Note: fs.create_feature_view exists but failed due to SDK incompatibility or signature mismatch:", msg)
                        print("Suggestion: Create the feature view manually in the Hopsworks UI or enable HOPS_AUTO_CREATE_FEATURE_VIEW=1 if your SDK supports programmatic creation.")
                        return None

                    # Attempt: positional (name, description)
                    try:
                        fv = fs.create_feature_view(view_name, description)
                        print(f"✓ Created feature view '{view_name}' via fs.create_feature_view(name, description)")
                        return fv
                    except Exception as e2:
                        if HOPS_DEBUG:
                            print(f"Note: fs.create_feature_view attempts failed: {e1} / {e2}")
                        else:
                            print("Note: fs.create_feature_view attempts failed; see HOPS_DEBUG=1 for details")
                        # continue to next creation strategies
            except Exception as e:
                print(f"Note: Unexpected error while trying fs.create_feature_view: {e}")

        # 3) Try creating via Feature Group object
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=1)
            if hasattr(fg, "create_feature_view"):
                try:
                    fv = fg.create_feature_view(name=view_name, description=description)
                    print(f"✓ Created feature view '{view_name}' via fg.create_feature_view()")
                    return fv
                except Exception as e:
                    msg = str(e)
                    if "featurestore_id" in msg or "FeatureView.__init__" in msg:
                        print("Note: fg.create_feature_view failed due to SDK constructor/signature changes requiring a 'featurestore_id'.")
                        print("Suggestion: Create the Feature View manually in Hopsworks UI or adapt this helper for your SDK version.")
                    elif HOPS_DEBUG:
                        print(f"Note: fg.create_feature_view failed: {e}")
                    else:
                        print("Note: fg.create_feature_view failed; enable HOPS_DEBUG=1 for more details")
        except Exception as e:
            print(f"Note: Could not access feature group while attempting to create view: {e}")

        # 4) Try constructing a FeatureView object (HSFS style) and pass it to fs.create_feature_view
        try:
            try:
                from hsfs.feature_view import FeatureView
            except Exception:
                FeatureView = None

            fg = None
            try:
                fg = fs.get_feature_group(name=feature_group_name, version=1)
            except Exception:
                pass

            if FeatureView is not None:
                try:
                    if fg is not None:
                        fv_obj = FeatureView(name=view_name, description=description, query=f"SELECT * FROM {feature_group_name}")
                    else:
                        fv_obj = FeatureView(name=view_name, description=description, query=f"SELECT * FROM {feature_group_name}")

                    try:
                        fv = fs.create_feature_view(fv_obj)
                        print(f"✓ Created feature view '{view_name}' via FeatureView object + fs.create_feature_view(obj)")
                        return fv
                    except Exception as e:
                        msg = str(e)
                        if "featurestore_id" in msg or "FeatureView.__init__" in msg:
                            print("Note: create_feature_view(FeatureView) failed due to SDK changes (e.g., FeatureView requires additional constructor args like featurestore_id).")
                            print("Suggestion: Create the feature view manually in Hopsworks UI or update this helper to match your HSFS SDK version.")
                            return None
                        if HOPS_DEBUG:
                            print(f"Note: create_feature_view(FeatureView) failed: {e}")
                        else:
                            print("Note: create_feature_view(FeatureView) failed; enable HOPS_DEBUG=1 for more details")
                except Exception as e:
                    if HOPS_DEBUG:
                        print(f"Note: could not build FeatureView object: {e}")
                    else:
                        print("Note: could not build FeatureView object; enable HOPS_DEBUG=1 for more details")
        except Exception:
            pass

        # If we reach here, we couldn't create the view programmatically
        print("Note: Could not programmatically create feature view '{0}' for feature group '{1}'.".format(view_name, feature_group_name))
        print("Suggested manual workaround:")
        print(f"  1) In Hopsworks UI, go to Feature Views -> New Feature View")
        print(f"  2) Name the view: {view_name}")
        print(f"  3) Use this SQL as a simple query: SELECT * FROM {feature_group_name}")
        print("  4) Add the created feature view to your pipelines or grant read permissions as needed.")
        return None

    except Exception as e:
        print(f"Unexpected error creating feature view: {e}")
        return None


def get_features_from_hopsworks(
    feature_group_name: str = "aqi_features",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> pd.DataFrame:
    """
    Retrieve features from Hopsworks Feature Store with robust fallbacks.

    The preferred path uses the Hopsworks Feature Query Service (fast, online) via
    `feature_view.read()`. If that fails (common when backend Hudi/HDFS files are
    missing or unavailable), the function attempts a fallback read using
    `fg.read()` which may read from the feature group's offline storage.

    Args:
        feature_group_name: Name of the feature group
        start_time: Start timestamp (ISO format)
        end_time: End timestamp (ISO format)

    Returns:
        DataFrame with features
    """
    if not HOPSWORKS_AVAILABLE:
        raise ImportError("Hopsworks is not installed")

    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
    except Exception as e:
        print(f"Error initializing Hopsworks project/feature store: {e}")
        raise

    try:
        fg = fs.get_feature_group(name=feature_group_name, version=1)
    except Exception as e:
        msg = f"Could not locate feature group '{feature_group_name}' in Hopsworks: {e}"
        print(msg)
        raise RuntimeError(msg) from e

    # Build the feature view / query
    try:
        if start_time and end_time:
            feature_view = fg.select_all().filter(
                (fg.timestamp >= start_time) & (fg.timestamp <= end_time)
            )
        else:
            feature_view = fg.select_all()

        # Primary attempt: use the Feature Query Service with retries/backoff for Hudi materialization
            retry_max = int(os.getenv("HOPS_QUERY_MAX_RETRIES", "6"))  # exponential backoff attempts
            base_delay = float(os.getenv("HOPS_QUERY_BASE_DELAY", "5.0"))  # seconds
            last_exc = None
            for attempt in range(1, retry_max + 1):
                try:
                    df = feature_view.read()
                    print(f"✓ Retrieved {len(df)} rows from feature group '{feature_group_name}' (query service)")
                    return df
                except Exception as qs_err:
                    last_exc = qs_err
                    # Detect specific server-side binder/schema errors which are NOT transient
                    err_text_full = str(qs_err)
                    err_text = err_text_full.lower()

                    # Case A: Hudi/HDFS/materialization errors (transient) -> retry
                    if "hoodie" in err_text or "hudi" in err_text or "hdfs" in err_text or "could not read data" in err_text:
                        delay = base_delay * (2 ** (attempt - 1))
                        print(f"Warning: Query Service read attempt {attempt}/{retry_max} failed with Hudi/HDFS error: {qs_err}")
                        print(f"Waiting {delay:.1f}s before retrying (exponential backoff)...")
                        time.sleep(delay)
                        continue

                    # Case B: Binder/DuckDB schema errors indicating malformed feature descriptors
                    if ("binder error" in err_text) or ("referenced column" in err_text) or ("{\'name\':" in err_text_full) or ("{'name':" in err_text_full):
                        print("Critical: Query Service returned a Binder/DuckDB error that likely indicates malformed feature group schema metadata:", qs_err)
                        print("Running schema inspection to gather diagnostic information...")
                        try:
                            schema_report = inspect_featuregroup_schema(feature_group_name)
                            print("Schema inspection result:")
                            print(schema_report)
                            if schema_report.get("issues"):
                                print("Detected schema issues that likely cause binder errors. Consider recreating the feature group or contacting your Hopsworks administrator.")
                                print("To attempt an automated recreate (DESTRUCTIVE), set environment variable FORCE_RECREATE_FEATURE_GROUP=1 and re-run ingestion or call attempt_recreate_featuregroup().")
                        except Exception as insp_exc:
                            print("Schema inspection failed:", insp_exc)

                        # These errors won't be fixed by retrying; abort immediately with guidance
                        raise RuntimeError(
                            "Hopsworks Query Service failed with a binder/DuckDB schema error. "
                            "This typically means the stored feature descriptors for the feature group are malformed. "
                            "Run inspect_featuregroup_schema() to collect diagnostics, and either recreate the feature group or contact Hopsworks support. "
                            f"Original server error: {qs_err}") from qs_err

                    # Non-Hudi, non-binder error - break and try offline fallback immediately
                    print("Error reading via Feature Query Service (non-Hudi error):", qs_err)
                    break

            # If we reach here, Query Service retries exhausted or non-Hudi error occurred
            print("Attempting fallback read via offline feature group storage (fg.read())...")
            try:
                df = fg.read()
                print(f"✓ Fallback succeeded, retrieved {len(df)} rows from feature group '{feature_group_name}' (offline read)")
                return df
            except Exception as offline_err:
                print("Fallback offline read also failed:", offline_err)

                # If we see binder/DuckDB messages here, run schema inspection and abort with guidance
                oe_text = str(offline_err)
                if ("binder error" in oe_text.lower()) or ("referenced column" in oe_text.lower()) or ("{'name':" in oe_text):
                    print("Detected Binder/DuckDB error in offline read. Running schema inspection...")
                    try:
                        schema_report = inspect_featuregroup_schema(feature_group_name)
                        print("Schema inspection result:")
                        print(schema_report)
                        if schema_report.get("issues"):
                            print("Detected schema issues that likely cause binder errors. Consider recreating the feature group or contacting your Hopsworks administrator.")
                            print("To attempt an automated recreate (DESTRUCTIVE), set environment variable FORCE_RECREATE_FEATURE_GROUP=1 and re-run ingestion or call attempt_recreate_featuregroup().")
                    except Exception as insp_exc:
                        print("Schema inspection failed:", insp_exc)
                    raise RuntimeError("Offline read failed due to a Binder/DuckDB schema error. See schema inspection output above for details.") from offline_err

                # Optional: wait for materialization if configured via environment vars
                wait_flag = os.getenv("HOPS_WAIT_FOR_MATERIALIZATION", "0").lower() in ("1", "true", "yes")
                if wait_flag:
                    max_wait = float(os.getenv("HOPS_MATERIALIZATION_MAX_WAIT", "600"))  # seconds
                    poll = float(os.getenv("HOPS_MATERIALIZATION_POLL", "10"))  # seconds
                    waited = 0.0
                    print(f"Info: Waiting up to {max_wait}s for feature group materialization (poll every {poll}s)...")
                    while waited < max_wait:
                        try:
                            sample = fg.select_all().limit(1).read()
                            if sample is not None and len(sample) > 0:
                                print("✓ Feature group is now readable after waiting; returning data.")
                                return fg.read()
                        except Exception:
                            # still not materialized
                            pass
                        time.sleep(poll)
                        waited += poll
                        print(f"Waiting for materialization... {waited:.0f}/{max_wait:.0f}s")
                    print("Timed out waiting for feature group materialization.")

                msg = (
                    "Could not read data using Hopsworks Query Service or fallback fg.read().\n"
                    "Possible causes: the feature group is not materialized in the backend storage (Hudi/HDFS),\n"
                    "there was an intermittent server error, or permissions are missing.\n"
                    "Suggested actions: re-run feature ingestion (features_extraction.py) to ensure data is persisted,\n"
                    "check Hopsworks project and HDFS/Hudi availability, and confirm you have read permissions.\n"
                    f"Original Query Service error (last): {last_exc}"
                )
                print(msg)
                raise RuntimeError(msg) from last_exc

    except Exception as e:
        print(f"Error retrieving from Hopsworks: {e}")
        raise


def create_training_dataset(
    feature_group_name: str = "aqi_features",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> pd.DataFrame:
    """
    Create training dataset from feature store.

    This function normalizes common Hopsworks sanitizations (e.g., lower-casing
    feature names) and ensures the returned DataFrame contains the expected
    columns used by the training pipeline (notably `AQI` and `AQI_Category`).

    Args:
        feature_group_name: Name of the feature group
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        DataFrame ready for training
    """
    df = get_features_from_hopsworks(feature_group_name, start_time, end_time)

    if df is None:
        raise RuntimeError("No data returned from Hopsworks feature group")

    # Normalize column names: detect lower-cased columns from Hopsworks and map them
    cols_lower_map = {c.lower(): c for c in df.columns}

    # Map 'aqi' -> 'AQI' if Hopsworks sanitized to lower case
    if 'aqi' in cols_lower_map and 'AQI' not in df.columns:
        orig = cols_lower_map['aqi']
        df['AQI'] = df[orig]
        print(f"Note: Mapped feature store column '{orig}' -> 'AQI' to match training expectations")

    # Map 'aqi_category' -> 'AQI_Category'
    if 'aqi_category' in cols_lower_map and 'AQI_Category' not in df.columns:
        orig = cols_lower_map['aqi_category']
        df['AQI_Category'] = df[orig]
        print(f"Note: Mapped feature store column '{orig}' -> 'AQI_Category' to match training expectations")

    # Ensure timestamp column exists and is datetime
    if 'timestamp' in cols_lower_map and 'timestamp' not in df.columns:
        # Use the original-cased timestamp column if present under different case
        orig_ts = cols_lower_map['timestamp']
        df['timestamp'] = pd.to_datetime(df[orig_ts])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Final validation: AQI must be present for supervised training
    if 'AQI' not in df.columns:
        raise ValueError("AQI column not found in features after normalization. Check feature group contents.")

    return df


def save_model_to_hopsworks(
    model_path,
    model_name: str,
    description: str = "",
    model_type: str = "sklearn",
    metadata: dict = None
):
    """
    Save model to Hopsworks Model Registry (Compatible with Hopsworks 4.2.10)
    
    Uses the correct framework-specific API (sklearn, tensorflow, torch, python)
    and uploads model artifacts as a directory.
    
    Args:
        model_path: Path to the model file (str or Path)
        model_name: Name for the model in the registry
        description: Optional description
        model_type: Framework type ('sklearn', 'tensorflow', 'torch', 'python')
        metadata: Optional metrics dict (e.g., {'accuracy': 0.95})
    
    Returns:
        True if upload was successful, False otherwise.
    """
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
    
    # ------------------------------------------------------------------
    # 1️⃣ Prepare model directory (REQUIRED by Hopsworks)
    # ------------------------------------------------------------------
    model_dir = model_path.parent / f"{model_name}_artifact"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model file to directory
    if model_path.is_file():
        shutil.copy(model_path, model_dir / model_path.name)
    else:
        # If it's already a directory, copy its contents
        shutil.copytree(model_path, model_dir, dirs_exist_ok=True)
    
    print(f"✓ Prepared model directory: {model_dir}")
    
    # ------------------------------------------------------------------
    # 2️⃣ Create model using framework-specific API
    # ------------------------------------------------------------------
    try:
        # Select the correct framework API
        framework_map = {
            'sklearn': mr.sklearn,
            'tensorflow': mr.tensorflow,
            'torch': mr.torch,
            'python': mr.python
        }
        
        if model_type not in framework_map:
            print(f"Warning: Unknown model_type '{model_type}', defaulting to 'python'")
            framework_api = mr.python
        else:
            framework_api = framework_map[model_type]
        
        # Filter metrics to only include numeric values (Hopsworks requirement)
        numeric_metrics = {}
        non_numeric_info = {}
        
        if metadata:
            for key, value in metadata.items():
                try:
                    # Try to convert to float - if successful, it's a valid metric
                    float(value)
                    numeric_metrics[key] = value
                except (ValueError, TypeError):
                    # Store non-numeric values for description
                    non_numeric_info[key] = value
        
        # Add non-numeric info to description
        full_description = description
        if non_numeric_info:
            info_str = ", ".join([f"{k}: {v}" for k, v in non_numeric_info.items()])
            full_description = f"{description} | {info_str}" if description else info_str
        
        # Create model metadata object with numeric metrics only
        model = framework_api.create_model(
            name=model_name,
            description=full_description,
            metrics=numeric_metrics
        )
        print(f"✓ Created model metadata for '{model_name}'")
        
        # ------------------------------------------------------------------
        # 3️⃣ Save model artifacts (this uploads to registry AND creates version)
        # ------------------------------------------------------------------
        model.save(str(model_dir))
        print(f"✓ Model '{model_name}' successfully uploaded to Hopsworks!")
        print(f"   Version: {model.version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading model to Hopsworks: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary directory
        try:
            if model_dir.exists():
                shutil.rmtree(model_dir)
        except Exception:
            pass


def load_model_from_hopsworks(
    model_name: str,
    version: Optional[int] = None,
    download_path: Optional[Path] = None,
    model_type: str = "sklearn"
):
    """
    Load a model from Hopsworks Model Registry.
    
    Args:
        model_name: Name of the model in the registry
        version: Optional version number (defaults to latest)
        download_path: Optional path to download model to (defaults to temp directory)
        model_type: Framework type ('sklearn', 'tensorflow', 'torch', 'python')
    
    Returns:
        Tuple of (model, version_number, model_path) or (None, None, None) on error
    """
    if not HOPSWORKS_AVAILABLE:
        print("Hopsworks not available. Cannot load model from registry.")
        return None, None, None
    
    import tempfile
    import joblib
    from pathlib import Path
    
    try:
        project = get_hopsworks_project()
        mr = project.get_model_registry()
        
        # Select the correct framework API
        framework_map = {
            'sklearn': mr.sklearn,
            'tensorflow': mr.tensorflow,
            'torch': mr.torch,
            'python': mr.python
        }
        
        if model_type not in framework_map:
            print(f"Warning: Unknown model_type '{model_type}', defaulting to 'python'")
            framework_api = mr.python
        else:
            framework_api = framework_map[model_type]
        
        # Get model
        if version is not None:
            model_meta = framework_api.get_model(name=model_name, version=version)
        else:
            model_meta = framework_api.get_model(name=model_name)
        
        print(f"✓ Found model '{model_name}' version {model_meta.version} in Hopsworks")
        
        # Download model
        if download_path is None:
            download_path = Path(tempfile.mkdtemp())
        else:
            download_path = Path(download_path)
            download_path.mkdir(parents=True, exist_ok=True)
        
        # Download model artifacts
        model_dir = download_path / f"{model_name}_v{model_meta.version}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_meta.download(str(model_dir))
        print(f"✓ Downloaded model to: {model_dir}")
        
        # Load the model file (look for .pkl files)
        model_file = None
        for pkl_file in model_dir.rglob("*.pkl"):
            model_file = pkl_file
            break
        
        if model_file is None:
            # Try to load from directory directly (for some frameworks)
            model = framework_api.load_model(model_name, version=model_meta.version if version is None else version)
            return model, model_meta.version, str(model_dir)
        else:
            # Load using joblib
            model = joblib.load(model_file)
            return model, model_meta.version, str(model_file)
        
    except Exception as e:
        print(f"❌ Error loading model from Hopsworks: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# Example usage
if __name__ == "__main__":
    # This is a simple diagnostics CLI for development use
    import argparse
    parser = argparse.ArgumentParser(description="Hopsworks Feature Store helper CLI")
    parser.add_argument("command", choices=["check", "inspect", "recreate", "create_view"], help="Command to run: check (materialization), inspect (schema), recreate (destructive, requires env var), create_view (ensure a feature view exists)")
    parser.add_argument("--fg", default="aqi_features", help="Feature group name (default: aqi_features)")
    parser.add_argument("--sample", action="store_true", help="For inspect: also attempt to read a small sample")
    parser.add_argument("--view", default=None, help="Optional feature view name to create (defaults to <fg>_view)")
    args = parser.parse_args()

    try:
        if args.command == "check":
            print("Checking feature group materialization...")
            print(check_featuregroup_materialized(args.fg))
        elif args.command == "inspect":
            print(f"Inspecting feature group schema for '{args.fg}'...")
            report = inspect_featuregroup_schema(args.fg)
            print(report)
            if args.sample and report.get("ok"):
                try:
                    project = get_hopsworks_project()
                    fs = project.get_feature_store()
                    fg = fs.get_feature_group(name=args.fg, version=1)
                    print("Sample read (limit 5):")
                    print(fg.select_all().limit(5).read())
                except Exception as e:
                    print("Could not read sample:", e)
        elif args.command == "recreate":
            print("Attempting to recreate (DESTRUCTIVE) feature group. Ensure FORCE_RECREATE_FEATURE_GROUP=1 is set in environment.")
            # Attempt to load a local sample to recreate - require user to provide a local CSV path via env var
            sample_path = os.getenv("RECREATE_SAMPLE_PATH")
            if not sample_path:
                print("Aborting: set RECREATE_SAMPLE_PATH to a CSV file path containing a representative sample of the feature group schema to recreate.")
            else:
                try:
                    df = pd.read_csv(sample_path)
                    result = attempt_recreate_featuregroup(args.fg, df)
                    print(result)
                except Exception as e:
                    print("Failed to recreate feature group:", e)
        elif args.command == "create_view":
            view_name = args.view or os.getenv("FEATURE_VIEW_NAME") or f"{args.fg}_view"
            print(f"Ensuring feature view '{view_name}' exists for feature group '{args.fg}'...")
            try:
                res = create_feature_view(args.fg, view_name=view_name)
                if res:
                    print(f"✓ Feature view '{view_name}' is ready")
                else:
                    print(f"Note: Could not programmatically create feature view '{view_name}'. Please create it manually in Hopsworks UI.")
            except Exception as e:
                print("Error creating feature view:", e)
    except Exception as e:
        print("Error running CLI command:", e)
