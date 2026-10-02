from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "patient-discharge",
    bootstrap_servers="10.47.122.7:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="patient-consumer-group-test",
    value_deserializer=lambda x: x.decode("utf-8")
)

print("========================================")
print("PATIENT READMISSION KAFKA CONSUMER")
print("========================================")
print("Connected to Kafka")
print("Topic: patient-discharge")
print("Waiting for patient records...")
print()

for message in consumer:
    print(
        f"Received record | "
        f"Partition: {message.partition} | "
        f"Offset: {message.offset}"
    )
    print(message.value)
    print("----------------------------------------")