import json
import subprocess
from pathlib import Path

from kafka import KafkaConsumer


KAFKA_SERVER = "localhost:9092"
TOPIC = "patient-discharge"
GROUP_ID = "patient-consumer-hdfs-100-test"
RECORD_LIMIT = 100

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_OUTPUT = BASE_DIR / "data" / "patient-discharge-100.jsonl"
HDFS_OUTPUT = "/patient-readmission/raw/patient-discharge-100.jsonl"


def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        key_deserializer=lambda key: key.decode("utf-8") if key else None,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    print(f"Listening to topic '{TOPIC}'...")
    print(f"Consumer group: {GROUP_ID}")
    print(f"Record limit: {RECORD_LIMIT}")
    print(f"Local output: {LOCAL_OUTPUT}")
    print(f"HDFS output: {HDFS_OUTPUT}")

    received = 0

    try:
        with LOCAL_OUTPUT.open("w", encoding="utf-8") as output_file:
            for message in consumer:
                record = {
                    "kafka_key": message.key,
                    "partition": message.partition,
                    "offset": message.offset,
                    "data": message.value,
                }

                output_file.write(json.dumps(record) + "\n")
                received += 1

                if received % 100 == 0:
                    print(f"Received {received} records")

                if received >= RECORD_LIMIT:
                    break
    finally:
        consumer.close()

    print(f"Kafka consumption complete. Total records received: {received}")

    subprocess.run(
        [
            "C:\hadoop\bin\hdfs.cmd",
            "dfs",
            "-put",
            "-f",
            str(LOCAL_OUTPUT),
            HDFS_OUTPUT,
        ],
        check=True,
    )

    print(f"Uploaded {LOCAL_OUTPUT.name} to HDFS.")
    print(f"HDFS output: {HDFS_OUTPUT}")


if __name__ == "__main__":
    main()
