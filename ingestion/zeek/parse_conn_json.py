
import json
import ipaddress

from pathlib import Path
from datetime import datetime, timezone


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/zeek-json/conn.log"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data/processed/zeek-normalized"
)

REJECTED_DIR = PROJECT_ROOT / "data/rejected"


# Required fields in Zeek connection records
REQUIRED_FIELDS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.resp_h",
    "proto"
]


def validate_ip(address):
    """Validate an IPv4 or IPv6 address."""
    ipaddress.ip_address(address)
    return address


def normalize_record(record):
    """Validate and standardize one Zeek connection record."""

    for field in REQUIRED_FIELDS:
        if field not in record or record[field] in (None, "-"):
            raise ValueError(f"Missing required field: {field}")

    timestamp = float(record["ts"])

    if timestamp < 0:
        raise ValueError("Timestamp cannot be negative")

    source_ip = validate_ip(record["id.orig_h"])
    destination_ip = validate_ip(record["id.resp_h"])

    timestamp_utc = datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).isoformat()

    return {
        "ts_epoch": timestamp,
        "ts_datetime_utc": timestamp_utc,
        "uid": record["uid"],
        "source_ip": source_ip,
        "source_port_or_type": record.get("id.orig_p"),
        "destination_ip": destination_ip,
        "destination_port_or_code": record.get("id.resp_p"),
        "protocol": record["proto"],
        "service": record.get("service"),
        "duration_seconds": record.get("duration"),
        "original_bytes": record.get("orig_bytes"),
        "response_bytes": record.get("resp_bytes"),
        "connection_state": record.get("conn_state")
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    valid_file = OUTPUT_DIR / "conn_normalized.jsonl"
    rejected_file = REJECTED_DIR / "conn_rejected.jsonl"

    total_records = 0
    valid_records = 0
    rejected_records = 0

    with (
        INPUT_FILE.open("r", encoding="utf-8") as source,
        valid_file.open("w", encoding="utf-8") as valid,
        rejected_file.open("w", encoding="utf-8") as rejected
    ):

        for line_number, line in enumerate(source, start=1):

            if not line.strip():
                continue

            total_records += 1

            try:
                record = json.loads(line)

                normalized = normalize_record(record)

                valid.write(
                    json.dumps(normalized) + "\n"
                )

                valid_records += 1

            except (json.JSONDecodeError, ValueError, TypeError, OverflowError) as error:

                rejected.write(
                    json.dumps({
                        "line_number": line_number,
                        "error": str(error),
                        "raw_record": line.strip()
                    }) + "\n"
                )

                rejected_records += 1

    print("\nNETSENTRY AI - INGESTION SUMMARY")
    print("--------------------------------")
    print(f"Total records:    {total_records}")
    print(f"Valid records:    {valid_records}")
    print(f"Rejected records: {rejected_records}")
    print(f"Valid output:     {valid_file}")
    print(f"Rejected output:  {rejected_file}")


if __name__ == "__main__":
    main()
