
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data/raw/cic-ids2017/extracted/MachineLearningCVE"
)

CSV_FILES = sorted(DATASET_DIR.glob("*.csv"))
CHUNK_SIZE = 100_000


def audit_file(file_path):
    print("\n" + "=" * 90)
    print(f"FILE: {file_path.name}")
    print("=" * 90)

    total_rows = 0
    label_counts = Counter()
    null_counts = Counter()
    infinite_counts = Counter()
    normalized_columns = None
    original_columns = None

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig"
    ):
        total_rows += len(chunk)

        # Preserve original headers from the first chunk.
        if original_columns is None:
            original_columns = list(chunk.columns)

        # Normalize whitespace in column names.
        chunk.columns = chunk.columns.str.strip()

        if normalized_columns is None:
            normalized_columns = list(chunk.columns)

        # Count missing values.
        null_counts.update(chunk.isna().sum().to_dict())

        # Count infinite values in numeric columns.
        numeric_data = chunk.select_dtypes(include="number")

        if not numeric_data.empty:
            infinite_values = np.isinf(numeric_data.to_numpy()).sum(axis=0)

            for column, count in zip(
                numeric_data.columns,
                infinite_values
            ):
                infinite_counts[column] += int(count)

        # Count labels while preserving their original meaning.
        if "Label" in chunk.columns:
            labels = chunk["Label"].astype("string").str.strip()
            label_counts.update(
                labels.fillna("<MISSING_LABEL>").value_counts().to_dict()
            )

    print(f"\nTOTAL ROWS: {total_rows:,}")
    print(f"TOTAL COLUMNS: {len(normalized_columns)}")

    print("\nORIGINAL HEADER EXAMPLES:")
    for original, normalized in zip(
        original_columns[:10],
        normalized_columns[:10]
    ):
        print(f"{original!r} -> {normalized!r}")

    print("\nLABEL DISTRIBUTION:")
    for label, count in label_counts.most_common():
        print(f"{label}: {count:,}")

    print("\nMISSING VALUES:")
    missing_found = False

    for column, count in sorted(null_counts.items()):
        if count > 0:
            print(f"{column}: {count:,}")
            missing_found = True

    if not missing_found:
        print("No missing values found.")

    print("\nINFINITE NUMERICAL VALUES:")
    infinite_found = False

    for column, count in sorted(infinite_counts.items()):
        if count > 0:
            print(f"{column}: {count:,}")
            infinite_found = True

    if not infinite_found:
        print("No infinite values found.")

    return normalized_columns


def main():
    print("NETSENTRY AI - CIC-IDS2017 FULL DATASET AUDIT")
    print("=" * 90)
    print(f"CSV files found: {len(CSV_FILES)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if not CSV_FILES:
        print("ERROR: No CSV files found!")
        return

    schemas = {}

    for file_path in CSV_FILES:
        schemas[file_path.name] = audit_file(file_path)

    print("\n" + "=" * 90)
    print("SCHEMA CONSISTENCY CHECK")
    print("=" * 90)

    reference_name = next(iter(schemas))
    reference_schema = schemas[reference_name]

    for filename, schema in schemas.items():
        if schema == reference_schema:
            print(f"MATCH: {filename}")
        else:
            print(f"MISMATCH: {filename}")

            print("Missing columns:", sorted(
                set(reference_schema) - set(schema)
            ))

            print("Extra columns:", sorted(
                set(schema) - set(reference_schema)
            ))

            if set(reference_schema) == set(schema):
                print("Column order differs.")

    print("\nFULL DATASET AUDIT COMPLETED")


if __name__ == "__main__":
    main()
