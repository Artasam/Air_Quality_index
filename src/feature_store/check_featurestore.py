"""
Diagnostic script to check what's in the Hopsworks feature store
Run this to understand the 91 vs 96 row discrepancy
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.feature_store.hopsworks_integration import get_features_from_hopsworks, get_hopsworks_project

def check_feature_store():
    """Check what's actually in the feature store"""
    
    print("="*70)
    print("FEATURE STORE DIAGNOSTIC CHECK")
    print("="*70)
    
    try:
        # Get ALL records without time filtering
        print("\n1️⃣  Fetching ALL records from feature store (no time filter)...")
        all_records = get_features_from_hopsworks(
            feature_group_name="aqi_features",
            start_time=None,
            end_time=None
        )
        
        if all_records is not None and len(all_records) > 0:
            print(f"   ✓ Total records in feature store: {len(all_records)}")
            
            # Normalize timestamp
            if 'timestamp' in all_records.columns:
                all_records['timestamp'] = pd.to_datetime(all_records['timestamp'])
            elif 'Timestamp' in all_records.columns:
                all_records['timestamp'] = pd.to_datetime(all_records['Timestamp'])
            
            if 'timestamp' in all_records.columns:
                print(f"   ✓ Date range: {all_records['timestamp'].min()} to {all_records['timestamp'].max()}")
                
                # Show distribution by date
                print("\n   Hourly distribution:")
                hourly_counts = all_records.groupby(all_records['timestamp'].dt.floor('H')).size()
                for ts, count in hourly_counts.items():
                    print(f"      {ts}: {count} records")
        else:
            print("   ⚠️  No records found in feature store")
        
        # Now fetch with time filter (like the script does)
        print("\n2️⃣  Fetching with time filter (2026-01-19 to 2026-01-22)...")
        filtered_records = get_features_from_hopsworks(
            feature_group_name="aqi_features",
            start_time="2026-01-19 00:00:00",
            end_time="2026-01-22 23:00:00"
        )
        
        if filtered_records is not None and len(filtered_records) > 0:
            print(f"   ✓ Filtered records: {len(filtered_records)}")
            
            if 'timestamp' in filtered_records.columns:
                filtered_records['timestamp'] = pd.to_datetime(filtered_records['timestamp'])
            elif 'Timestamp' in filtered_records.columns:
                filtered_records['timestamp'] = pd.to_datetime(filtered_records['Timestamp'])
            
            if 'timestamp' in filtered_records.columns:
                print(f"   ✓ Date range: {filtered_records['timestamp'].min()} to {filtered_records['timestamp'].max()}")
        else:
            print("   ⚠️  No records found with time filter")
        
        # Compare
        if all_records is not None and filtered_records is not None:
            print(f"\n3️⃣  Comparison:")
            print(f"   Total records: {len(all_records)}")
            print(f"   Filtered records: {len(filtered_records)}")
            print(f"   Difference: {len(all_records) - len(filtered_records)} records")
            
            if len(all_records) > len(filtered_records):
                print(f"\n   ⚠️  {len(all_records) - len(filtered_records)} records are outside the time filter!")
                
                # Find which timestamps are missing
                if 'timestamp' in all_records.columns and 'timestamp' in filtered_records.columns:
                    all_ts = set(all_records['timestamp'])
                    filtered_ts = set(filtered_records['timestamp'])
                    missing_ts = all_ts - filtered_ts
                    
                    if missing_ts:
                        print(f"\n   Missing timestamps (outside filter range):")
                        for ts in sorted(missing_ts)[:20]:
                            print(f"      - {ts}")
        
        # Check for duplicates
        print(f"\n4️⃣  Checking for duplicates...")
        if all_records is not None and len(all_records) > 0 and 'timestamp' in all_records.columns:
            duplicates = all_records.duplicated(subset=['timestamp'], keep=False)
            dup_count = duplicates.sum()
            if dup_count > 0:
                print(f"   ⚠️  Found {dup_count} duplicate timestamps!")
                print("\n   Duplicate timestamps:")
                dup_records = all_records[duplicates].sort_values('timestamp')
                for ts in dup_records['timestamp'].unique()[:10]:
                    count = len(dup_records[dup_records['timestamp'] == ts])
                    print(f"      {ts}: {count} copies")
            else:
                print(f"   ✓ No duplicates found")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_feature_store()