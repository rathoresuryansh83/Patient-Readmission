import csv
import json
from pathlib import Path

from kafka import KafkaProducer


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "kafka_data.csv"

KAFKA_SERVER = "localhost:9092"
TOPIC = "patient-discharge"

# Set to None to send the complete dataset.
RECORD_LIMIT = 100

PROGRESS_EVERY = 100


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

    sent = 0

    print(f"Reading: {INPUT_FILE}")
    print(f"Kafka server: {KAFKA_SERVER}")
    print(f"Topic: {TOPIC}")
    print(f"Record limit: {RECORD_LIMIT}")

    try:
        with INPUT_FILE.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                encounter_id = row["encounter_id"]

                producer.send(
                    TOPIC,
                    key=encounter_id,
                    value=row,
                )

                sent += 1

                if sent % PROGRESS_EVERY == 0:
                    producer.flush()
                    print(f"Sent {sent} records")

                if RECORD_LIMIT is not None and sent >= RECORD_LIMIT:
                    break

        producer.flush()

    finally:
        producer.close()

    print(f"Producer complete. Total records sent: {sent}")


if __name__ == "__main__":
    main()
