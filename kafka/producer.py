import csv
import json
from pathlib import Path

from kafka import KafkaProducer


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "kafka_data.csv"

KAFKA_SERVER = "localhost:9092"
TOPIC = "patient-discharge"
TEST_RECORDS = 5


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        key_serializer=lambda key: str(key).encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5,
    )


def main():
    producer = create_producer()

    print(f"Reading: {INPUT_FILE}")
    print(f"Sending first {TEST_RECORDS} records to '{TOPIC}'")

    with INPUT_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for count, row in enumerate(reader, start=1):
            encounter_id = row["encounter_id"]

            future = producer.send(
                TOPIC,
                key=encounter_id,
                value=row,
            )

            metadata = future.get(timeout=30)

            print(
                f"Sent record {count}: "
                f"encounter_id={encounter_id}, "
                f"partition={metadata.partition}, "
                f"offset={metadata.offset}"
            )

            if count >= TEST_RECORDS:
                break

    producer.flush()
    producer.close()

    print("Producer test complete.")


if __name__ == "__main__":
    main()
