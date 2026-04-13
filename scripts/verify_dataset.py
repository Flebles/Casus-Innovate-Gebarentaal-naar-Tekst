#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path


# Verify and summarize collected gesture dataset
def verify_dataset(dataset_path: str = "data/gestures.csv"):

    path = Path(dataset_path)

    if not path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        print("\nRun collection first:")
        print("  python main_collect.py --gesture 1 --samples 50")
        return False

    print(f"\nDataset Verification: {dataset_path}")
    print("="*60)

    df = pd.read_csv(dataset_path)

    # Display overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total samples:     {len(df)}")
    print(f"  Total gestures:    {df['gesture'].nunique()}")
    print(f"  Total columns:     {len(df.columns)}")

    print(f"\nPer-Gesture Breakdown:")
    gesture_counts = df['gesture'].value_counts().sort_index()

    for gesture, count in gesture_counts.items():
        status = "OK" if count >= 50 else "LOW"
        print(f"  [{status}] {gesture:15s}: {count:3d} samples")

    print(f"\nData Quality Checks:")

    # Check for missing values
    null_cols = df.isnull().sum()
    if null_cols.sum() > 0:
        print(f"  [WARNING] Missing values detected:")
        for col, count in null_cols[null_cols > 0].items():
            print(f"    - {col}: {count} missing")
    else:
        print(f"  [OK] No missing values")

    # Check if gesture column exists
    if 'gesture' in df.columns:
        print(f"  [OK] 'gesture' column present")
    else:
        print(f"  [ERROR] 'gesture' column missing!")

    # Check for landmark columns
    landmark_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z', 'visibility'))]
    if len(landmark_cols) >= 84:
        print(f"  [OK] Landmark columns: {len(landmark_cols)} found")
    else:
        print(f"  [WARNING] Landmark columns: only {len(landmark_cols)} found (expected ~84+)")

    print(f"\nRecommendations:")

    min_samples = gesture_counts.min()
    max_samples = gesture_counts.max()

    if min_samples < 30:
        print(f"  [WARNING] Some gestures have <30 samples (collect more for better accuracy)")
        print(f"    Minimum: {min_samples} | Target: 50")
    elif min_samples < 50:
        print(f"  [WARNING] Some gestures have <50 samples (consider collecting more)")
        print(f"    Minimum: {min_samples} | Target: 50")
    else:
        print(f"  [OK] All gestures have sufficient samples (>=50)")

    if df['gesture'].nunique() < 5:
        print(f"  [WARNING] Only {df['gesture'].nunique()} unique gestures (target: 5)")
    else:
        print(f"  [OK] Good gesture diversity ({df['gesture'].nunique()} unique)")

    print(f"\nReady to train!")
    print(f"  Run: python main_train.py --dataset {dataset_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify collected gesture dataset"
    )
    parser.add_argument(
        "--dataset",
        default="data/gestures.csv",
        help="Path to dataset CSV"
    )

    args = parser.parse_args()
    verify_dataset(args.dataset)

