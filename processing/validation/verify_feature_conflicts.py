
from pathlib import Path
import sqlite3
from collections import defaultdict

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

# Retrieve only fingerprints previously identified as conflicts.
connection = sqlite3.connect(DB_PATH)

candidate_hashes = {
    row[0]
    for row in connection.execute("""
        SELECT feature_hash
        FROM feature_labels
        GROUP BY feature_hash
        HAVING COUNT(*) > 1
    """)
}

connection.close()

print(f"Candidate fingerprints: {len(candidate_hashes):,}")

# fingerprint -> actual feature tuple -> set of labels
verified_groups = defaultdict(lambda: defaultdict(set))

total_matching_rows = 0
reference_features = None

files = sorted(DATA_DIR.glob("*_cleaned.csv"))

for file_path in files:
    print(f"\nScanning: {file_path.name}")

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
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

        hashes = pd.util.hash_pandas_object(
            chunk[feature_columns],
            index=False
        )

        # Restrict detailed comparisons to candidate fingerprints.
        mask = hashes.astype(str).isin(candidate_hashes)

        if not mask.any():
            continue

        matching_features = chunk.loc[mask, feature_columns]
        matching_hashes = hashes.loc[mask]
        matching_labels = chunk.loc[mask, "label"]

        for row, fingerprint, label in zip(
            matching_features.itertuples(index=False, name=None),
            matching_hashes,
            matching_labels
        ):
            # All feature columns were converted to numeric
            # during cleaning. Normalize missing values.
            exact_feature_tuple = tuple(
                None if pd.isna(value) else float(value)
                for value in row
            )

            fingerprint = str(int(fingerprint))

            verified_groups[fingerprint][exact_feature_tuple].add(
                str(label)
            )

            total_matching_rows += 1

    print(f"Completed: {file_path.name}")

# Count exact feature tuples that have multiple labels.
exact_conflicts = []

hash_collision_buckets = 0

for fingerprint, feature_groups in verified_groups.items():

    if len(feature_groups) > 1:
        hash_collision_buckets += 1

    for feature_tuple, labels in feature_groups.items():
        if len(labels) > 1:
            exact_conflicts.append({
                "fingerprint": fingerprint,
                "labels": sorted(labels),
            })

print("\n" + "=" * 65)
print("EXACT FEATURE CONFLICT VERIFICATION")
print("=" * 65)

print(f"Candidate fingerprints: {len(candidate_hashes):,}")
print(f"Matching rows examined: {total_matching_rows:,}")
print(f"Hash buckets with multiple actual feature tuples: {hash_collision_buckets:,}")
print(f"Exact feature groups with conflicting labels: {len(exact_conflicts):,}")

if exact_conflicts:
    print("\nExamples of verified exact conflicts:")

    for conflict in exact_conflicts[:20]:
        print(
            f"{conflict['fingerprint']}: "
            f"{', '.join(conflict['labels'])}"
        )

print("\nVERIFICATION COMPLETED")
