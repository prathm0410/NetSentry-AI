
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/cic-ids2017/cleaned"
DB_PATH = ROOT / "data/processed/cic-ids2017/feature_label_audit.sqlite"

CHUNK_SIZE = 100_000

metadata_columns = {
    "source_file",
    "source_row_number",
    "source_label",
    "label",
}

if DB_PATH.exists():
    DB_PATH.unlink()

connection = sqlite3.connect(DB_PATH)

connection.execute("""
    CREATE TABLE feature_labels (
        feature_hash TEXT NOT NULL,
        label TEXT NOT NULL,
        PRIMARY KEY (feature_hash, label)
    )
""")

total_rows = 0
reference_features = None

try:
    files = sorted(DATA_DIR.glob("*_cleaned.csv"))

    if len(files) != 8:
        raise RuntimeError(
            f"Expected 8 cleaned files, found {len(files)}"
        )

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

        file_rows = 0

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE
        ):
            feature_columns = [
                col for col in chunk.columns
                if col not in metadata_columns
            ]

            if reference_features is None:
                reference_features = feature_columns
            elif feature_columns != reference_features:
                raise ValueError(
                    f"Feature schema mismatch: {file_path.name}"
                )

            if chunk["label"].isna().any():
                raise ValueError(
                    f"Missing labels found in {file_path.name}"
                )

            # Hash features only. Labels are stored separately.
            feature_hashes = pd.util.hash_pandas_object(
                chunk[feature_columns],
                index=False
            )

            records = (
                (str(int(feature_hash)), str(label))
                for feature_hash, label in zip(
                    feature_hashes,
                    chunk["label"]
                )
            )

            connection.executemany(
                """
                INSERT OR IGNORE INTO feature_labels
                (feature_hash, label)
                VALUES (?, ?)
                """,
                records
            )

            file_rows += len(chunk)
            total_rows += len(chunk)

            print(
                f"Rows processed: {file_rows:,}",
                end="\r"
            )

        connection.commit()
        print(f"Completed: {file_rows:,} rows")

    unique_feature_hashes = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT feature_hash
            FROM feature_labels
            GROUP BY feature_hash
        )
    """).fetchone()[0]

    conflicting_groups = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT feature_hash
            FROM feature_labels
            GROUP BY feature_hash
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    print("\n" + "=" * 65)
    print("FEATURE-LEVEL LABEL CONFLICT AUDIT")
    print("=" * 65)

    print(f"Total rows scanned: {total_rows:,}")
    print(f"Unique feature hashes: {unique_feature_hashes:,}")
    print(f"Conflicting feature groups: {conflicting_groups:,}")

    if conflicting_groups:
        print("\nExamples of conflicting feature groups:")

        examples = connection.execute("""
            SELECT feature_hash, GROUP_CONCAT(label, ', ')
            FROM feature_labels
            GROUP BY feature_hash
            HAVING COUNT(*) > 1
            LIMIT 20
        """).fetchall()

        for feature_hash, labels in examples:
            print(f"{feature_hash}: {labels}")

    else:
        print("\nNo conflicting labels detected.")

    print("\nAUDIT COMPLETED")

finally:
    connection.close()
