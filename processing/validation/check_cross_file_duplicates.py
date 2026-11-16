
import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data/raw/cic-ids2017/extracted/MachineLearningCVE"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = OUTPUT_DIR / "cic_duplicate_index.sqlite"

CHUNK_SIZE = 100_000


def main():
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS seen_rows (
            row_hash TEXT NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (row_hash, source_file)
        ) WITHOUT ROWID
    """)

    connection.commit()

    files = sorted(DATASET_DIR.glob("*.csv"))

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig"
        ):
            chunk.columns = chunk.columns.str.strip()

            hashes = pd.util.hash_pandas_object(
                chunk,
                index=False
            )

            records = [
                (str(int(value)), file_path.name)
                for value in hashes
            ]

            connection.executemany("""
                INSERT OR IGNORE INTO seen_rows
                (row_hash, source_file)
                VALUES (?, ?)
            """, records)

            connection.commit()

    unique_cross_file_rows = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT row_hash
            FROM seen_rows
            GROUP BY row_hash
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    extra_file_occurrences = connection.execute("""
        SELECT COALESCE(SUM(file_count - 1), 0)
        FROM (
            SELECT COUNT(*) AS file_count
            FROM seen_rows
            GROUP BY row_hash
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    print("\n" + "=" * 70)
    print("CROSS-FILE DUPLICATE AUDIT")
    print("=" * 70)

    print(
        "Unique full-row hashes appearing in multiple files:",
        f"{unique_cross_file_rows:,}"
    )

    print(
        "Additional file appearances of those rows:",
        f"{extra_file_occurrences:,}"
    )

    print("\nCROSS-FILE AUDIT COMPLETED")

    connection.close()


if __name__ == "__main__":
    main()
