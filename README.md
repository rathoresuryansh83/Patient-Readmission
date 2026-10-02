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
## Machine Learning and Prediction

The ML pipeline now includes preprocessing, Logistic Regression training, test-set prediction generation, and readmission probability output.

- Training rows: 81,412
- Testing rows: 20,354
- Transformed features: 2,370
- Accuracy: 0.6423
- Precision: 0.1660
- Recall: 0.5482
- F1-score: 0.2548
- ROC-AUC: 0.6441

Prediction output is stored locally at `data/ml/predictions.csv` and verified in HDFS at:

```text
/patient-readmission/predictions/predictions.csv
```

The HDFS prediction file contains 20,354 prediction records plus one header row. Each record contains the actual 30-day readmission label, predicted label, and readmission probability.

The prediction generation script is located at `ml/predict.py`.
