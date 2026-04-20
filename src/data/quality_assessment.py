"""
CSV Dataset Consistency Checker

Algorithm:
    1. Feature Column Consistency - Verify both datasets have identical columns.
    2. Column Order - Verify columns appear in the exact same order.
    3. Missing Values Check - Detect NaN/empty values, flag suspicious zero-filled columns.
    4. Data Types Check - Verify mineral_name is string, feature columns are numeric.

Author: Nathaniel de la Rosa
"""

import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE1 = PROJECT_ROOT / "data" / "processed" / "measured_preprocessed.csv"
FILE2 = PROJECT_ROOT / "data" / "processed" / "synthetic_preprocessed.csv"
LABEL_COL = "mineral_name"
SEPARATOR = "─" * 40


def load_datasets(file1: str, file2: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load two CSV files into pandas DataFrames."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    assert isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame)
    return df1, df2


def check_column_consistency(df1: pd.DataFrame, df2: pd.DataFrame, file1: str, file2: str) -> None:
    """Verify both datasets have identical columns with no extras or missing."""
    print(SEPARATOR)
    print("1. FEATURE COLUMN CONSISTENCY")
    print(SEPARATOR)

    set1, set2 = set(df1.columns), set(df2.columns)
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    if not only_in_1 and not only_in_2 and len(df1.columns) == len(df2.columns):
        print(f"✅ Columns MATCH — {len(df1.columns)} identical columns in both datasets.")
    else:
        print("❌ Column mismatch detected.")
        print(f"   {file1}: {len(df1.columns)} columns, {file2}: {len(df2.columns)} columns")
        if only_in_1:
            print(f"   Only in {file1}: {sorted(only_in_1)}")
        if only_in_2:
            print(f"   Only in {file2}: {sorted(only_in_2)}")


def check_column_order(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Verify columns appear in the exact same order across both datasets."""
    print("\n" + SEPARATOR)
    print("2. COLUMN ORDER")
    print(SEPARATOR)

    cols1, cols2 = list(df1.columns), list(df2.columns)

    if cols1 == cols2:
        print("✅ Column order is identical.")
    else:
        print("❌ Column order differs:")
        for i, (c1, c2) in enumerate(zip(cols1, cols2)):
            if c1 != c2:
                print(f"   Position {i + 1}: '{c1}' vs '{c2}'")


def check_missing_values(df1: pd.DataFrame, df2: pd.DataFrame, file1: str, file2: str) -> None:
    """Check for NaN/empty values and flag suspicious entirely-zero columns."""
    print("\n" + SEPARATOR)
    print("3. MISSING VALUES CHECK")
    print(SEPARATOR)

    for label, df in [(file1, df1), (file2, df2)]:
        nan_counts = df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]

        if nan_cols.empty:
            print(f"✅ {label}: No missing values.")
            continue

        print(f"⚠️  {label}: Missing values found:")
        for col, count in nan_cols.items():
            print(f"   '{col}': {count} missing")

        filled_with_zero = (df[nan_cols.index] == 0).all()
        zero_filled_cols = filled_with_zero[filled_with_zero].index.tolist()
        if zero_filled_cols:
            print(f"   ℹ️  Zero-filled columns: {zero_filled_cols}")

        feature_cols = [c for c in df.columns if c != LABEL_COL]
        all_zero_cols = [c for c in feature_cols if (df[c] == 0).all()]
        if all_zero_cols:
            print(f"   ❌ Suspicious — entirely zero columns: {all_zero_cols}")


def check_label_column(col: pd.Series) -> None:
    """Validate mineral_name is categorical string with no corrupted characters."""
    if col.dtype == object or pd.api.types.is_string_dtype(col):
        non_string_vals = col.dropna()
        non_string_vals = non_string_vals[non_string_vals.apply(lambda x: not isinstance(x, str))]

        if non_string_vals.empty:
            print(f"   ✅ '{LABEL_COL}' is categorical (string) — correct.")
        else:
            print(f"   ❌ '{LABEL_COL}' contains non-string values: {sorted(set(non_string_vals.tolist()))}")

        suspicious = col[col.apply(lambda x: isinstance(x, str) and bool(re.search(r'[^\x00-\x7F]', x)))]
        if suspicious.empty:
            print(f"   ✅ All mineral names are clean (no suspicious characters).")
        else:
            print(f"   ⚠️  Mineral names with special characters (verify if corrupted): {sorted(suspicious.unique().tolist())}")
    else:
        print(f"   ❌ '{LABEL_COL}' is numeric (dtype: {col.dtype}) — appears already encoded, expected raw strings.")


def check_data_types(df1: pd.DataFrame, df2: pd.DataFrame, file1: str, file2: str) -> None:
    """Verify mineral_name is string and all feature columns are numeric."""
    print("\n" + SEPARATOR)
    print("4. DATA TYPES CHECK")
    print(SEPARATOR)

    for label, df in [(file1, df1), (file2, df2)]:
        print(f"\n{label}:")
        feature_cols = [c for c in df.columns if c != LABEL_COL]

        if LABEL_COL not in df.columns:
            print(f"   ❌ '{LABEL_COL}' column not found.")
        else:
            check_label_column(df[LABEL_COL])

        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        int_cols = [c for c in feature_cols if df[c].dtype == 'int64']

        if not non_numeric:
            print(f"   ✅ All feature columns are numeric.")
        else:
            print(f"   ❌ Non-numeric feature columns: {non_numeric}")

        if int_cols:
            print(f"   ⚠️  Integer (int64) columns found (expected float64): {int_cols}")
        else:
            print(f"   ✅ All feature columns are float64.")


def main() -> None:
    """Entry point. Loads datasets and runs all consistency checks in sequence."""
    df1, df2 = load_datasets(FILE1, FILE2)

    NAME1 = FILE1.name
    NAME2 = FILE2.name

    check_column_consistency(df1, df2, NAME1, NAME2)
    check_column_order(df1, df2)
    check_missing_values(df1, df2, NAME1, NAME2)
    check_data_types(df1, df2, NAME1, NAME2)

    print("\n" + SEPARATOR)
    print("CHECK COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()