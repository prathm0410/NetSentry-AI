
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data/raw/cic-ids2017/extracted/MachineLearningCVE"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/cic-ids2017/cleaned"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data/processed/cic-ids2017/cleaning_report.json"
)

CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LABEL NORMALIZATION
# --------------------------------------------------

def normalize_label(label):
    """
    Preserve the original label separately.
    Convert labels into consistent ML-friendly names.
    """

    if pd.isna(label):
        return "UNKNOWN"

    label = str(label).strip()

    # Replace corrupted Unicode replacement character
    # with a separator, without modifying source_label.
    label = label.replace("\ufffd", " ")

    # Normalize case and punctuation.
    label = label.upper()
    label = re.sub(r"[^A-Z0-9]+", "_", label)
    label = label.strip("_")

    return label if label else "UNKNOWN"


# --------------------------------------------------
# 3. CLEAN ONE CSV FILE IN CHUNKS
# --------------------------------------------------

def clean_file(file_path):

    print("\n" + "=" * 80)
    print(f"PROCESSING: {file_path.name}")
    print("=" * 80)

    output_file = OUTPUT_DIR / f"{file_path.stem}_cleaned.csv"

    if output_file.exists():
        output_file.unlink()

    stats = {
        "source_file": file_path.name,
        "output_file": output_file.name,
        "total_rows": 0,
        "missing_values": Counter(),
        "infinite_values": Counter(),
        "numeric_conversion_failures": Counter(),
        "labels": Counter(),
        "original_columns": [],
        "normalized_columns": [],
    }

    first_chunk = True
    row_offset = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
        ),
        start=1,
    ):

        # ------------------------------------------
        # A. Normalize column names
        # ------------------------------------------

        original_columns = list(chunk.columns)

        chunk.columns = chunk.columns.str.strip()

        if len(set(chunk.columns)) != len(chunk.columns):
            raise ValueError(
                f"Duplicate columns after normalization: "
                f"{file_path.name}"
            )

        if "Label" not in chunk.columns:
            raise ValueError(
                f"Label column missing in {file_path.name}"
            )

        if first_chunk:
            stats["original_columns"] = original_columns
            stats["normalized_columns"] = list(chunk.columns)

        # ------------------------------------------
        # B. Preserve original labels
        # ------------------------------------------

        source_labels = chunk["Label"].astype("string").str.strip()

        chunk = chunk.drop(columns=["Label"])

        chunk["source_label"] = source_labels

        chunk["label"] = source_labels.map(normalize_label)

        # ------------------------------------------
        # C. Convert features to numeric values
        # ------------------------------------------

        feature_columns = [
            column
            for column in chunk.columns
            if column not in ("source_label", "label")
        ]

        for column in feature_columns:

            original_values = chunk[column]

            converted = pd.to_numeric(
                original_values,
                errors="coerce",
            )

            # Count values that could not be converted.
            failed = original_values.notna() & converted.isna()

            stats["numeric_conversion_failures"][column] += int(
                failed.sum()
            )

            chunk[column] = converted

        # ------------------------------------------
        # D. Count missing values
        # ------------------------------------------

        missing_counts = chunk[feature_columns].isna().sum()

        for column, count in missing_counts.items():
            stats["missing_values"][column] += int(count)

        # ------------------------------------------
        # E. Count and replace infinite values
        # ------------------------------------------

        numeric_features = chunk[feature_columns].select_dtypes(
            include="number"
        )

        if not numeric_features.empty:

            infinite_mask = pd.DataFrame(
                np.isinf(numeric_features.to_numpy()),
                columns=numeric_features.columns,
                index=numeric_features.index,
            )

            infinite_counts = infinite_mask.sum()

            for column, count in infinite_counts.items():
                stats["infinite_values"][column] += int(count)

            # Convert positive and negative infinity to NaN.
            chunk[feature_columns] = chunk[
                feature_columns
            ].replace([np.inf, -np.inf], np.nan)

        # ------------------------------------------
        # F. Preserve source information
        # ------------------------------------------

        row_count = len(chunk)

        chunk.insert(
            0,
            "source_file",
            file_path.name,
        )

        chunk.insert(
            1,
            "source_row_number",
            range(row_offset + 1, row_offset + row_count + 1),
        )

        row_offset += row_count

        # ------------------------------------------
        # G. Track labels and row counts
        # ------------------------------------------

        stats["total_rows"] += row_count

        stats["labels"].update(
            chunk["label"].value_counts().to_dict()
        )

        # ------------------------------------------
        # H. Write cleaned chunk to disk
        # ------------------------------------------

        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False

        print(
            f"Chunk {chunk_number} processed | "
            f"Rows processed: {stats['total_rows']:,}"
        )

    print(f"\nRows cleaned: {stats['total_rows']:,}")
    print(f"Output: {output_file}")

    # Convert Counters into normal dictionaries for JSON.
    for key in (
        "missing_values",
        "infinite_values",
        "numeric_conversion_failures",
        "labels",
    ):
        stats[key] = dict(stats[key])

    return stats


# --------------------------------------------------
# 4. PROCESS ALL EIGHT CSV FILES
# --------------------------------------------------

def main():

    print("NETSENTRY AI - CIC-IDS2017 CLEANING PIPELINE")
    print("=" * 80)

    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_DIR}"
        )

    print(f"Input files found: {len(csv_files)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    all_stats = []

    for file_path in csv_files:
        stats = clean_file(file_path)
        all_stats.append(stats)

    # ----------------------------------------------
    # 5. SAVE CLEANING REPORT
    # ----------------------------------------------

    report = {
        "project": "NetSentry AI",
        "dataset": "CIC-IDS2017",
        "files_processed": len(all_stats),
        "total_rows_processed": sum(
            item["total_rows"] for item in all_stats
        ),
        "cleaning_policy": {
            "column_names": "Whitespace stripped",
            "original_labels": "Preserved as source_label",
            "normalized_labels": "Stored as label",
            "missing_values": "Preserved as NaN",
            "infinite_values": "Converted to NaN",
            "duplicates": "Preserved",
            "rare_classes": "Preserved",
            "source_information": "Filename and row number retained",
        },
        "files": all_stats,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("CLEANING PIPELINE COMPLETED")
    print("=" * 80)

    print(f"Files processed: {len(all_stats)}")

    print(
        "Total rows processed:",
        f"{report['total_rows_processed']:,}",
    )

    print(f"Cleaning report: {REPORT_PATH}")
    print(f"Cleaned data directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
