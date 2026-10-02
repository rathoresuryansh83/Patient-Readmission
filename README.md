# Patient Readmission Prediction using Kafka, HDFS, Machine Learning and Dashboard

## Project Overview

This project builds an end-to-end patient readmission prediction pipeline using:

- Python
- Apache Kafka
- HDFS
- Machine Learning
- Dashboard interface

The project uses the UCI Diabetes 130-US Hospitals dataset and processes patient encounter records through a local Kafka and HDFS environment.

The project is currently configured to run on **one Windows PC**, with Kafka and HDFS running locally.

---

## Architecture

```text
Diabetes 130-US Hospitals Dataset
                |
                v
        Data Preparation
                |
                v
          kafka_data.csv
                |
                v
       Kafka Producer
                |
                v
     Kafka Topic: patient-discharge
                |
                v
       Kafka Consumer
                |
                v
             HDFS
                |
                v
      ML Preprocessing
                |
                v
        Model Training
                |
                v
       Readmission Prediction
                |
                v
       Risk / Prediction Output
                |
                v
       Dashboard Interface