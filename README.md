# Patient Readmission Prediction using Kafka and Machine Learning

## Project Overview

This project builds a patient readmission prediction pipeline using the Diabetes 130-US Hospitals dataset, Apache Kafka, and machine learning.

The project is developed and tested on one Windows PC.

## Architecture

Dataset -> Data Preparation -> Kafka Producer -> patient-discharge -> Kafka Consumer -> HDFS -> ML Pipeline -> Predictions/Risk -> Dashboard Interface

## Current Environment

- Windows
- Python 3.14.7
- Apache Kafka 4.3.1
- Java 17.0.20.101
- kafka-python 3.0.11
- Kafka broker: localhost:9092

## Dataset

- 101,766 hospital encounters
- 50 original columns
- 71,518 unique patients
- encounter_id is unique per encounter

The original CSV files are kept locally and excluded from GitHub.

## Data Preparation

data/prepare_data.py:

1. Loads the raw dataset.
2. Converts ? to missing values.
3. Creates readmitted_30_days.
4. Creates patient_id.
5. Preserves encounter-level records.
6. Creates cleaned_data.csv.
7. Creates kafka_data.csv without the target.

Target:

readmitted_30_days = 1 when readmitted == <30
readmitted_30_days = 0 otherwise

Current target distribution:

- 0: 90,409
- 1: 11,357

## Kafka

Kafka runs locally at localhost:9092.

Topic: patient-discharge

- Partitions: 1
- Replication factor: 1

## Kafka Producer

kafka/producer.py reads data/kafka_data.csv and sends JSON encounter records to Kafka using encounter_id as the message key.

A 5-record producer test has been completed successfully.

## Kafka Consumer

kafka/consumer.py consumes records from patient-discharge.

A 5-record Producer -> Kafka -> Consumer test has been completed successfully.

## Project Status

### Completed

- [x] Kafka installation verified
- [x] Java 17 configured
- [x] Fresh Kafka storage initialized
- [x] Kafka broker running
- [x] Kafka topic created
- [x] Dataset copied locally
- [x] Dataset inspected
- [x] Initial data preparation completed
- [x] Kafka producer created
- [x] Kafka consumer created
- [x] 5-record end-to-end Kafka test completed

### Next

- [ ] Finalize producer and consumer
- [ ] Full Kafka data flow
- [ ] HDFS storage
- [ ] ML preprocessing
- [ ] Model training and evaluation
- [ ] Prediction output
- [ ] Dashboard interface contract

## Team Responsibilities

### Person 1

Dataset, initial cleaning, target creation, Kafka producer.

### Person 2

Kafka consumer and HDFS.

### Person 3

ML preprocessing, model training, and predictions.

### Person 4

Dashboard.

The dashboard itself is not implemented in this local setup.

## Repository Structure

patient-readmission/
├── data/
│   ├── diabetic_data.csv
│   ├── cleaned_data.csv
│   ├── kafka_data.csv
│   └── prepare_data.py
├── kafka/
│   ├── producer.py
│   └── consumer.py
├── ml/
├── docs/
├── .gitignore
└── README.md
