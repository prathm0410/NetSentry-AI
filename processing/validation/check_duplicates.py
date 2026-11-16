
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data/raw/cic-ids2017/extracted/MachineLearningCVE"
)

CHUNK_SIZE = 100_000


def audit_duplicates(file_path):
    print(f"\nFILE: {file_path.name}")

    seen = set()
    total_rows = 0
    duplicate_rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig"
    ):
        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        # Hash complete rows, including their labels.
        row_hashes = pd.util.hash_pandas_object(
            chunk,
            index=False
        )

        for value in row_hashes:
            value = int(value)

            if value in seen:
                duplicate_rows += 1
            else:
                seen.add(value)

    print(f"Total rows: {total_rows:,}")
    print(f"Duplicate rows within file: {duplicate_rows:,}")


def main():
    for file_path in sorted(DATASET_DIR.glob("*.csv")):
        audit_duplicates(file_path)


if __name__ == "__main__":
    main()
