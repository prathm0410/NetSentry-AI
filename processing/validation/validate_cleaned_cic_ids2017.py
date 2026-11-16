
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/processed/cic-ids2017/cleaned"
REPORT_PATH = ROOT / "data/processed/cic-ids2017/cleaning_report.json"

CHUNK_SIZE = 100_000

report = json.loads(REPORT_PATH.read_text())
overall_rows = 0
overall_missing = 0
overall_infinite = 0
overall_null_labels = 0
overall_null_source_labels = 0

reference_columns = None
overall_labels = Counter()

print("=" * 70)
print("CIC-IDS2017 CLEANED DATA VALIDATION")
print("=" * 70)

for item in report["files"]:
    source_name = item["source_file"]
    expected_rows = item["total_rows"]

    output_name = Path(source_name).stem + "_cleaned.csv"
    output_path = DATA_DIR / output_name

    if not output_path.exists():
        raise FileNotFoundError(output_path)

    rows = 0
    missing = 0
    infinite = 0
    null_labels = 0
    null_source_labels = 0
    labels = Counter()
    first_row = None
    last_row = None

    for chunk in pd.read_csv(output_path, chunksize=CHUNK_SIZE):
        rows += len(chunk)

        # Verify schema consistency across all files.
        if reference_columns is None:
            reference_columns = list(chunk.columns)
        elif list(chunk.columns) != reference_columns:
            raise ValueError(f"Schema mismatch: {output_name}")

        # Check original and normalized labels.
        null_labels += int(chunk["label"].isna().sum())
        null_source_labels += int(chunk["source_label"].isna().sum())

        labels.update(chunk["label"].value_counts().to_dict())

        # Exclude metadata and labels from feature checks.
        metadata = {
            "source_file",
            "source_row_number",
            "source_label",
            "label",
        }
        feature_columns = [
            col for col in chunk.columns if col not in metadata
        ]

        features = chunk[feature_columns]

        missing += int(features.isna().sum().sum())

        numeric_features = features.select_dtypes(include="number")

        if not numeric_features.empty:
            infinite += int(
                np.isinf(numeric_features.to_numpy()).sum()
            )

        if first_row is None:
            first_row = int(chunk["source_row_number"].min())

        last_row = int(chunk["source_row_number"].max())

    overall_rows += rows
    overall_missing += missing
    overall_infinite += infinite
    overall_null_labels += null_labels
    overall_null_source_labels += null_source_labels
    overall_labels.update(labels)

    print(f"\nFile: {output_name}")
    print(f"Rows: {rows:,} / Expected: {expected_rows:,}")
    print(f"Row count correct: {rows == expected_rows}")
    print(f"Source row-number range: {first_row} to {last_row}")
    print(f"Missing feature values: {missing:,}")
    print(f"Infinite feature values: {infinite:,}")
    print(f"Missing normalized labels: {null_labels}")
    print(f"Missing original labels: {null_source_labels}")
    print(f"Classes: {dict(labels)}")

print("\n" + "=" * 70)
print("OVERALL VALIDATION SUMMARY")
print("=" * 70)

print(f"Total rows: {overall_rows:,}")
print(f"Total missing feature values: {overall_missing:,}")
print(f"Total infinite feature values: {overall_infinite:,}")
print(f"Missing normalized labels: {overall_null_labels}")
print(f"Missing original labels: {overall_null_source_labels}")
print(f"Unique normalized classes: {len(overall_labels)}")
print(f"Consistent columns: {len(reference_columns)}")

print("\nOverall class distribution:")
for label, count in overall_labels.most_common():
    print(f"{label}: {count:,}")

print("\nVALIDATION COMPLETED")
