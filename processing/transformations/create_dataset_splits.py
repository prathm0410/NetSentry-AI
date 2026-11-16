
from pathlib import Path
from collections import Counter
import gzip
import json
import sqlite3

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/processed/cic-ids2017/cleaned"
OUTPUT_DIR = ROOT / "data/processed/cic-ids2017/splits"

DB_PATH = ROOT / "data/processed/cic-ids2017/group_split_manifest.sqlite"
REPORT_PATH = ROOT / "data/processed/cic-ids2017/split_report.json"

CHUNK_SIZE = 100_000


PARTITIONS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}

METADATA_COLUMNS = {
    "source_file",
    "source_row_number",
    "source_label",
    "label",
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def assign_groups(connection, label, group_type, query, total_rows):
    """Assign complete feature groups to partitions."""

    targets = {
        name: total_rows * ratio
        for name, ratio in PARTITIONS.items()
    }

    assigned = {name: 0 for name in PARTITIONS}

    cursor = connection.execute(query, (label,) if label else ())

    batch = []

    for feature_hash, row_count in cursor:
        # Choose the partition with the greatest relative
        # remaining capacity for this class.
        selected = max(
            PARTITIONS,
            key=lambda name: (
                targets[name] - assigned[name]
            ) / targets[name]
        )

        assigned[selected] += row_count

        batch.append((
            feature_hash,
            selected,
            group_type,
            row_count,
        ))

        if len(batch) >= 10_000:
            connection.executemany("""
                INSERT INTO split_assignment
                (feature_hash, partition, group_type, row_count)
                VALUES (?, ?, ?, ?)
            """, batch)
            connection.commit()
            batch.clear()

    if batch:
        connection.executemany("""
            INSERT INTO split_assignment
            (feature_hash, partition, group_type, row_count)
            VALUES (?, ?, ?, ?)
        """, batch)
        connection.commit()

    print(f"\n{group_type}: {label or 'ALL CONFLICTS'}")
    for name in PARTITIONS:
        print(
            f"  {name}: {assigned[name]:,} / "
            f"target {targets[name]:,.0f}"
        )


def get_split_mapping(connection, hashes):
    """Retrieve partition assignments for one CSV chunk."""

    unique_hashes = pd.unique(hashes).tolist()
    mapping = {}

    # Stay below SQLite's usual parameter limit.
    for start in range(0, len(unique_hashes), 500):
        batch = unique_hashes[start:start + 500]
        placeholders = ",".join("?" for _ in batch)

        rows = connection.execute(
            f"""
            SELECT feature_hash, partition
            FROM split_assignment
            WHERE feature_hash IN ({placeholders})
            """,
            batch,
        ).fetchall()

        mapping.update(rows)

    return mapping


# --------------------------------------------------
# INITIAL CHECKS
# --------------------------------------------------

files = sorted(DATA_DIR.glob("*_cleaned.csv"))

if len(files) != 8:
    raise RuntimeError(
        f"Expected 8 cleaned CSV files; found {len(files)}"
    )

if abs(sum(PARTITIONS.values()) - 1.0) > 1e-9:
    raise ValueError("Partition ratios must total 1.0")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("NETSENTRY AI - LEAKAGE-AWARE DATASET SPLITTING")
print("=" * 70)

# Remove only outputs from a previous run of this script.
for name in PARTITIONS:
    path = OUTPUT_DIR / f"{name}.csv.gz"
    if path.exists():
        path.unlink()

for path in (DB_PATH, REPORT_PATH):
    if path.exists():
        path.unlink()


# --------------------------------------------------
# PASS 1: BUILD CANONICAL FEATURE GROUPS
# --------------------------------------------------

connection = sqlite3.connect(DB_PATH)

connection.execute("""
    CREATE TABLE group_labels (
        feature_hash TEXT NOT NULL,
        label TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        PRIMARY KEY (feature_hash, label)
    )
""")

connection.execute("""
    CREATE TABLE split_assignment (
        feature_hash TEXT PRIMARY KEY,
        partition TEXT NOT NULL,
        group_type TEXT NOT NULL,
        row_count INTEGER NOT NULL
    )
""")

total_rows = 0
reference_features = None

try:
    print("\nPASS 1: Building feature groups")

    for file_path in files:
        print(f"\nScanning: {file_path.name}")

        header = pd.read_csv(file_path, nrows=0)
        feature_columns = [
            col for col in header.columns
            if col not in METADATA_COLUMNS
        ]

        if reference_features is None:
            reference_features = feature_columns
        elif feature_columns != reference_features:
            raise ValueError(
                f"Feature schema mismatch: {file_path.name}"
            )

        # Force consistent numeric representation in every chunk.
        feature_dtypes = {
            col: "float64" for col in feature_columns
        }

        file_rows = 0

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            dtype=feature_dtypes,
        ):
            if chunk["label"].isna().any():
                raise ValueError(
                    f"Missing labels in {file_path.name}"
                )

            hashes = pd.util.hash_pandas_object(
                chunk[feature_columns],
                index=False,
            )

            temp = pd.DataFrame({
                "feature_hash": hashes.map(
                    lambda value: str(int(value))
                ),
                "label": chunk["label"].astype(str).values,
            })

            grouped = (
                temp.groupby(
                    ["feature_hash", "label"],
                    sort=False,
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

            file_rows += len(chunk)
            total_rows += len(chunk)

        connection.commit()
        print(f"Completed: {file_rows:,} rows")

    print(f"\nTotal rows scanned: {total_rows:,}")


    # --------------------------------------------------
    # PASS 2: ASSIGN GROUPS TO PARTITIONS
    # --------------------------------------------------

    print("\nPASS 2: Assigning feature groups")

    class_rows = connection.execute("""
        SELECT label, SUM(row_count)
        FROM group_labels
        GROUP BY label
        ORDER BY label
    """).fetchall()

    # Single-label groups are stratified by their label.
    pure_group_query = """
        SELECT gl.feature_hash, gl.row_count
        FROM group_labels gl
        WHERE gl.label = ?
          AND (
              SELECT COUNT(*)
              FROM group_labels other
              WHERE other.feature_hash = gl.feature_hash
          ) = 1
        ORDER BY gl.row_count DESC, gl.feature_hash
    """

    for label, _ in class_rows:
        pure_total = connection.execute("""
            SELECT COALESCE(SUM(gl.row_count), 0)
            FROM group_labels gl
            WHERE gl.label = ?
              AND (
                  SELECT COUNT(*)
                  FROM group_labels other
                  WHERE other.feature_hash = gl.feature_hash
              ) = 1
        """, (label,)).fetchone()[0]

        if pure_total:
            assign_groups(
                connection,
                label,
                "single_label",
                pure_group_query,
                pure_total,
            )

    # Conflicting groups are assigned as complete groups,
    # without pretending they have a single correct label.
    conflict_total = connection.execute("""
        SELECT COALESCE(SUM(group_rows), 0)
        FROM (
            SELECT SUM(row_count) AS group_rows
            FROM group_labels
            GROUP BY feature_hash
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if conflict_total:
        conflict_query = """
            SELECT feature_hash, SUM(row_count) AS group_rows
            FROM group_labels
            GROUP BY feature_hash
            HAVING COUNT(*) > 1
            ORDER BY group_rows DESC, feature_hash
        """

        assign_groups(
            connection,
            None,
            "conflicting",
            conflict_query,
            conflict_total,
        )

    assigned_groups = connection.execute("""
        SELECT COUNT(*) FROM split_assignment
    """).fetchone()[0]

    expected_groups = connection.execute("""
        SELECT COUNT(DISTINCT feature_hash)
        FROM group_labels
    """).fetchone()[0]

    if assigned_groups != expected_groups:
        raise RuntimeError(
            f"Assignment mismatch: {assigned_groups:,} vs "
            f"{expected_groups:,} groups"
        )


    # --------------------------------------------------
    # PASS 3: WRITE COMPRESSED DATASETS
    # --------------------------------------------------

    print("\nPASS 3: Writing compressed split files")

    output_handles = {
        name: gzip.open(
            OUTPUT_DIR / f"{name}.csv.gz",
            "wt",
            newline="",
            encoding="utf-8",
        )
        for name in PARTITIONS
    }

    output_written = Counter()
    output_labels = {
        name: Counter() for name in PARTITIONS
    }

    try:
        for file_path in files:
            print(f"\nWriting: {file_path.name}")

            feature_columns = reference_features
            feature_dtypes = {
                col: "float64" for col in feature_columns
            }

            for chunk in pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
                dtype=feature_dtypes,
            ):
                hashes = pd.util.hash_pandas_object(
                    chunk[feature_columns],
                    index=False,
                ).map(lambda value: str(int(value)))

                mapping = get_split_mapping(
                    connection,
                    hashes,
                )

                split_values = hashes.map(mapping)

                if split_values.isna().any():
                    raise RuntimeError(
                        f"Missing group assignment in {file_path.name}"
                    )

                for name in PARTITIONS:
                    mask = split_values.eq(name)

                    if not mask.any():
                        continue

                    output_chunk = chunk.loc[mask]

                    output_chunk.to_csv(
                        output_handles[name],
                        index=False,
                        header=(
                            output_written[name] == 0
                        ),
                    )

                    output_written[name] += len(output_chunk)

                    output_labels[name].update(
                        output_chunk["label"].value_counts().to_dict()
                    )

    finally:
        for handle in output_handles.values():
            handle.close()


    # --------------------------------------------------
    # PASS 4: VALIDATE ASSIGNMENTS
    # --------------------------------------------------

    print("\nPASS 4: Validating split assignments")

    assignment_counts = connection.execute("""
        SELECT partition, COUNT(*), SUM(row_count)
        FROM split_assignment
        GROUP BY partition
    """).fetchall()

    assignment_summary = {
        name: {
            "groups": 0,
            "rows": 0,
        }
        for name in PARTITIONS
    }

    for name, groups, rows in assignment_counts:
        assignment_summary[name] = {
            "groups": groups,
            "rows": rows,
        }

    # Every group must have exactly one partition.
    duplicate_assignments = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT feature_hash
            FROM split_assignment
            GROUP BY feature_hash
            HAVING COUNT(DISTINCT partition) > 1
        )
    """).fetchone()[0]

    if duplicate_assignments != 0:
        raise RuntimeError(
            "Feature-group leakage detected in assignment manifest"
        )

    written_total = sum(output_written.values())

    if written_total != total_rows:
        raise RuntimeError(
            f"Output row mismatch: {written_total:,} vs {total_rows:,}"
        )

    report = {
        "allocation_strategy": "deterministic_greedy_by_descending_group_size",
        "partition_ratios": PARTITIONS,
        "total_rows": total_rows,
        "total_feature_groups": expected_groups,
        "conflicting_feature_groups": connection.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT feature_hash
                FROM group_labels
                GROUP BY feature_hash
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0],
        "duplicate_group_assignments": duplicate_assignments,
        "partitions": {},
    }

    for name in PARTITIONS:
        report["partitions"][name] = {
            "rows_written": output_written[name],
            "groups_assigned": assignment_summary[name]["groups"],
            "manifest_rows": assignment_summary[name]["rows"],
            "class_distribution": dict(output_labels[name]),
            "file": f"{name}.csv.gz",
        }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SPLIT GENERATION COMPLETED")
    print("=" * 70)

    print(f"Total input rows: {total_rows:,}")
    print(f"Total output rows: {written_total:,}")
    print(f"Total feature groups: {expected_groups:,}")
    print(f"Conflicting feature groups: {report['conflicting_feature_groups']:,}")
    print(f"Groups assigned to multiple partitions: {duplicate_assignments}")

    for name in PARTITIONS:
        print(
            f"\n{name.upper()}: "
            f"{output_written[name]:,} rows "
            f"({output_written[name] / total_rows:.2%})"
        )

        print(
            f"  Groups: "
            f"{assignment_summary[name]['groups']:,}"
        )

        print(
            f"  File: "
            f"{OUTPUT_DIR / (name + '.csv.gz')}"
        )

    print(f"\nReport: {REPORT_PATH}")
    print(f"Manifest: {DB_PATH}")

    print("\nSPLIT PIPELINE COMPLETED SUCCESSFULLY")

finally:
    connection.close()
