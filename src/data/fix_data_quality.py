"""
Programmer: Nathaniel de la Rosa

Fixes identified data quality issues in the preprocessed datasets:
1. Convert int64 feature columns to float64 in measured_preprocessed.csv
2. Fix corrupted mineral name characters in synthetic_preprocessed.csv
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEASURED_FILE = PROJECT_ROOT / "data" / "processed" / "measured_preprocessed.csv"
SYNTHETIC_FILE = PROJECT_ROOT / "data" / "processed" / "synthetic_preprocessed.csv"

CORRUPTED_NAMES = {
    "gaspã©ite": "gaspéite",
    "lã¶llingite": "lölllingite",
    "schã¤ferite": "schäferite",
}


def fix_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any int64 feature columns to float64."""
    int_cols = [c for c in df.columns if df[c].dtype == "int64"]
    df[int_cols] = df[int_cols].astype(float)
    print(f"   Converted {len(int_cols)} int64 column(s) to float64: {int_cols}")
    return df


def fix_corrupted_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace corrupted mineral name strings with correct UTF-8 versions."""
    df["mineral_name"] = df["mineral_name"].replace(CORRUPTED_NAMES)
    for corrupted, fixed in CORRUPTED_NAMES.items():
        print(f"   '{corrupted}' → '{fixed}'")
    return df


def main():
    # --- Fix measured ---
    print("Fixing measured_preprocessed.csv...")
    measured_df = pd.read_csv(MEASURED_FILE)
    measured_df = fix_int_columns(measured_df)
    measured_df.to_csv(MEASURED_FILE, index=False)
    print("   ✅ Saved.\n")

    # --- Fix synthetic ---
    print("Fixing synthetic_preprocessed.csv...")
    synthetic_df = pd.read_csv(SYNTHETIC_FILE)
    synthetic_df = fix_corrupted_names(synthetic_df)
    synthetic_df.to_csv(SYNTHETIC_FILE, index=False)
    print("   ✅ Saved.")


if __name__ == "__main__":
    main()