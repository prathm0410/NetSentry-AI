
from pathlib import Path
from collections import Counter
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data/processed/cic-ids2017/splits"
DB_PATH = SPLIT_DIR / "independent_validation.sqlite"

FILES = {
    "train": SPLIT_DIR / "train.csv.gz",
    "validation": SPLIT_DIR / "validation.csv.gz",
    "test": SPLIT_DIR / "test.csv.gz",
}

EXPECTED_COLUMNS = 82
EXPECTED_ROWS = 2_830_743
CHUNK_SIZE = 100_000

METADATA_COLUMNS = {
    "label",
    "source_label",
    "source_file",
    "source_row_number",
}

EXPECTED_CLASSES = {
    "BENIGN", "DOS_HULK", "PORTSCAN", "DDOS",
    "DOS_GOLDENEYE", "FTP_PATATOR", "SSH_PATATOR",
    "DOS_SLOWLORIS", "DOS_SLOWHTTPTEST", "BOT",
    "WEB_ATTACK_BRUTE_FORCE", "WEB_ATTACK_XSS",
    "WEB_ATTACK_SQL_INJECTION", "INFILTRATION",
    "HEARTBLEED",
}


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE seen_groups (
            feature_hash TEXT NOT NULL,
            partition TEXT NOT NULL,
            PRIMARY KEY (feature_hash, partition)
        )
    """)

    total_rows = 0
    total_groups = 0
    all_classes = set()
    reference_columns = None
    partition_rows = {}

    try:
        for partition, file_path in FILES.items():
            print(f"\nValidating {partition.upper()}...")

            if not file_path.exists():
                raise FileNotFoundError(file_path)

            rows = 0
            labels = Counter()
            groups_in_partition = 0

            for chunk in pd.read_csv(
                file_path,
                compression="gzip",
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):
                if reference_columns is None:
                    reference_columns = list(chunk.columns)

                    if len(reference_columns) != EXPECTED_COLUMNS:
                        raise ValueError(
                            f"Expected {EXPECTED_COLUMNS} columns, "
                            f"found {len(reference_columns)}"
                        )

                    missing_metadata = (
                        METADATA_COLUMNS - set(reference_columns)
                    )
                    if missing_metadata:
                        raise ValueError(
                            f"Missing metadata columns: {missing_metadata}"
                        )

                elif list(chunk.columns) != reference_columns:
                    raise ValueError(
                        f"Column mismatch in {partition}"
                    )

                if chunk["label"].isna().any():
                    raise ValueError(
                        f"Missing labels found in {partition}"
                    )

                labels.update(chunk["label"].value_counts().to_dict())
                all_classes.update(chunk["label"].unique())

                feature_columns = [
                    col for col in reference_columns
                    if col not in METADATA_COLUMNS
                ]

                # Match the split generator's float64 feature hashing.
                features = chunk[feature_columns].astype("float64")

                hashes = pd.util.hash_pandas_object(
                    features,
                    index=False,
                )

                unique_hashes = set(map(int, hashes.tolist()))
                groups_in_partition += len(unique_hashes)

                conn.executemany(
                    """
                    INSERT OR IGNORE INTO seen_groups
                    (feature_hash, partition)
                    VALUES (?, ?)
                    """,
                    (
                        (str(feature_hash), partition)
                        for feature_hash in unique_hashes
                    ),
                )

                rows += len(chunk)

            conn.commit()

            partition_rows[partition] = rows
            total_rows += rows

            print(f"Rows: {rows:,}")
            print(f"Distinct feature groups in partition: "
                  f"{groups_in_partition:,}")
            print("Class count:", len(labels))

            for label, count in sorted(labels.items()):
                print(f"  {label}: {count:,}")

        overlap_count = conn.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT feature_hash
                FROM seen_groups
                GROUP BY feature_hash
                HAVING COUNT(DISTINCT partition) > 1
            )
        """).fetchone()[0]

        total_groups = conn.execute("""
            SELECT COUNT(DISTINCT feature_hash)
            FROM seen_groups
        """).fetchone()[0]

        print("\n" + "=" * 65)
        print("INDEPENDENT SPLIT VALIDATION SUMMARY")
        print("=" * 65)

        print(f"Total rows: {total_rows:,}")
        print(f"Expected rows: {EXPECTED_ROWS:,}")
        print(f"Total distinct feature groups: {total_groups:,}")
        print(f"Cross-partition overlapping groups: {overlap_count}")
        print(f"Unique classes found: {len(all_classes)}")
        print(f"Expected classes: {len(EXPECTED_CLASSES)}")
        print(f"Missing classes: {sorted(EXPECTED_CLASSES - all_classes)}")
        print(f"Unexpected classes: {sorted(all_classes - EXPECTED_CLASSES)}")

        print("\nPartition row counts:")
        for name, count in partition_rows.items():
            print(f"{name.upper()}: {count:,}")

        checks = {
            "row_count": total_rows == EXPECTED_ROWS,
            "schema": (
                reference_columns is not None
                and len(reference_columns) == EXPECTED_COLUMNS
            ),
            "all_classes_present": all_classes == EXPECTED_CLASSES,
            "no_cross_partition_overlap": overlap_count == 0,
            "all_partitions_nonempty": all(
                count > 0 for count in partition_rows.values()
            ),
        }

        print("\nVALIDATION CHECKS:")
        for name, passed in checks.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")

        if all(checks.values()):
            print("\nALL INDEPENDENT VALIDATION CHECKS PASSED")
        else:
            print("\nVALIDATION FAILED — INVESTIGATE BEFORE TRAINING")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
