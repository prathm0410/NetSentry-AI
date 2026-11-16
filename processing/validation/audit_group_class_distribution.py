
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/cic-ids2017/cleaned"
DB_PATH = ROOT / "data/processed/cic-ids2017/group_class_audit.sqlite"

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
    CREATE TABLE group_labels (
        feature_hash TEXT NOT NULL,
        label TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        PRIMARY KEY (feature_hash, label)
    )
""")

total_rows = 0
reference_features = None

try:
    files = sorted(DATA_DIR.glob("*_cleaned.csv"))

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

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
                raise ValueError("Feature schema mismatch")

            hashes = pd.util.hash_pandas_object(
                chunk[feature_columns],
                index=False
            )

            temp = pd.DataFrame({
                "feature_hash": hashes.map(str),
                "label": chunk["label"].astype(str).values,
            })

            grouped = (
                temp.groupby(
                    ["feature_hash", "label"],
                    sort=False
                )
                .size()
                .reset_index(name="row_count")
            )

            connection.executemany("""
                INSERT INTO group_labels
                (feature_hash, label, row_count)
                VALUES (?, ?, ?)
                ON CONFLICT(feature_hash, label)
                DO UPDATE SET
                    row_count = row_count + excluded.row_count
            """, grouped.itertuples(index=False, name=None))

            total_rows += len(chunk)

        connection.commit()
        print(f"Completed: {file_path.name}")

    print("\n" + "=" * 65)
    print("GROUP-LEVEL CLASS DISTRIBUTION")
    print("=" * 65)

    results = connection.execute("""
        SELECT
            label,
            SUM(row_count) AS records,
            COUNT(*) AS distinct_feature_groups
        FROM group_labels
        GROUP BY label
        ORDER BY records DESC
    """).fetchall()

    print(f"Total records scanned: {total_rows:,}")
    print("\nClass | Records | Distinct feature groups")

    for label, records, groups in results:
        print(f"{label} | {records:,} | {groups:,}")

    total_groups = connection.execute("""
        SELECT COUNT(DISTINCT feature_hash)
        FROM group_labels
    """).fetchone()[0]

    conflicting_groups = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT feature_hash
            FROM group_labels
            GROUP BY feature_hash
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    print(f"\nTotal distinct feature groups: {total_groups:,}")
    print(f"Groups with conflicting labels: {conflicting_groups:,}")
    print("\nGROUP AUDIT COMPLETED")

finally:
    connection.close()
