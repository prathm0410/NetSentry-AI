
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data/raw/cic-ids2017/extracted/MachineLearningCVE"
)

CSV_FILES = sorted(DATASET_DIR.glob("*.csv"))


def inspect_file(file_path):
    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    try:
        # Read only five records initially
        df = pd.read_csv(
            file_path,
            nrows=5,
            encoding="utf-8-sig"
        )

        print(f"\nSample shape: {df.shape}")

        print("\nCOLUMNS:")
        for column in df.columns:
            print(repr(column))

        print("\nDATA TYPES:")
        print(df.dtypes.to_string())

        print("\nSAMPLE RECORDS:")
        print(df.head(3).to_string(index=False))

        print("\nLABEL SAMPLES:")
        label_columns = [
            column for column in df.columns
            if column.strip().lower() == "label"
        ]

        if label_columns:
            print(df[label_columns[0]].to_string(index=False))
        else:
            print("Label column not found!")

    except Exception as error:
        print(f"ERROR inspecting {file_path.name}: {error}")


def main():
    print("NETSENTRY AI - CIC-IDS2017 DATASET INSPECTION")
    print("=" * 80)

    print(f"Dataset directory: {DATASET_DIR}")
    print(f"CSV files found: {len(CSV_FILES)}")

    if not CSV_FILES:
        print("ERROR: No CSV files found!")
        return

    for file_path in CSV_FILES:
        inspect_file(file_path)

    print("\n" + "=" * 80)
    print("INITIAL DATASET INSPECTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
