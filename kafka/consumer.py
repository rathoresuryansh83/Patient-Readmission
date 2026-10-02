import json

from kafka import KafkaConsumer


KAFKA_SERVER = "localhost:9092"
TOPIC = "patient-discharge"
GROUP_ID = "patient-consumer-100-test"
RECORD_LIMIT = 100


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
    print("Waiting for Kafka records...")

    received = 0

    for message in consumer:
        received += 1

        print(
            f"Received record {received}: "
            f"key={message.key}, "
            f"partition={message.partition}, "
            f"offset={message.offset}"
        )

        print(f"encounter_id={message.value.get('encounter_id')}")
        print(f"patient_id={message.value.get('patient_id')}")

        if received >= RECORD_LIMIT:
            break

    consumer.close()

    print(f"Consumer test complete. Total records received: {received}")


if __name__ == "__main__":
    main()