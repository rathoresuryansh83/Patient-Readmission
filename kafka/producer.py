import csv
import json
import time
from pathlib import Path

from kafka import KafkaProducer


# ============================================================
# PATIENT READMISSION - KAFKA PRODUCER
# PERSON 1
# ============================================================

KAFKA_SERVER = "localhost:9092"
TOPIC = "patient-discharge"

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "kafka_data.csv"
)

# TEST MODE
NUM_RECORDS = 20

# Small delay for testing
MESSAGE_DELAY = 0.1


print("=" * 65)
print("PATIENT READMISSION KAFKA PRODUCER")
print("=" * 65)

print(f"Kafka server : {KAFKA_SERVER}")
print(f"Kafka topic  : {TOPIC}")
print(f"Data file    : {DATA_FILE}")
print(f"Records      : {NUM_RECORDS}")


# ============================================================
# STEP 1 - CHECK DATASET
# ============================================================

if not DATA_FILE.exists():

    print("\nERROR: kafka_data.csv was not found.")
    print(f"Expected: {DATA_FILE}")

    raise SystemExit(1)

print("\nDataset found successfully.")


# ============================================================
# STEP 2 - CONNECT TO KAFKA
# ============================================================

try:

    producer = KafkaProducer(

       bootstrap_servers="10.47.122.7:9092",

        key_serializer=lambda key:
            str(key).encode("utf-8"),

        value_serializer=lambda value:
            json.dumps(value).encode("utf-8"),

        acks="all",

        retries=5,

        request_timeout_ms=30000
    )

    print("Kafka producer connected successfully.")

except Exception as e:

    print("\nERROR: Could not connect to Kafka.")
    print(e)

    raise SystemExit(1)


# ============================================================
# STEP 3 - READ CSV
# ============================================================

sent_count = 0

try:

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:

            print("ERROR: CSV header not found.")

            raise SystemExit(1)

        print(
            f"CSV columns found: "
            f"{len(reader.fieldnames)}"
        )

        # ====================================================
        # STEP 4 - SEND RECORDS
        # ====================================================

        for row in reader:

            if sent_count >= NUM_RECORDS:
                break

            patient_id = row["patient_nbr"]

            future = producer.send(
                TOPIC,
                key=patient_id,
                value=dict(row)
            )

            metadata = future.get(timeout=30)

            sent_count += 1

            print(
                f"Sent record {sent_count:02d} | "
                f"Patient: {patient_id} | "
                f"Partition: {metadata.partition} | "
                f"Offset: {metadata.offset}"
            )

            time.sleep(MESSAGE_DELAY)

    producer.flush()

except Exception as e:

    print("\nERROR while sending records:")
    print(e)

finally:

    producer.close()


# ============================================================
# STEP 5 - FINAL RESULT
# ============================================================

print("\n" + "=" * 65)
print("PRODUCER COMPLETED")
print("=" * 65)

print(f"Records successfully sent: {sent_count}")

if sent_count == NUM_RECORDS:

    print("SUCCESS: All test records were sent.")

else:

    print("WARNING: Some records were not sent.")