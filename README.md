# Patient Readmission Risk Prediction using Kafka and Big Data

## 📌 Project Overview

The **Patient Readmission Risk Prediction System** is a Big Data and Machine Learning project designed to process patient healthcare data, identify factors associated with hospital readmission, and support data-driven discharge planning.

The project combines **Apache Kafka**, **Python**, **Machine Learning**, and **Big Data processing concepts** to create a pipeline for handling patient records.

The system processes patient information through a data pipeline, prepares the data for machine learning, and generates predictions that can be used to identify patients who may have a higher risk of readmission.

---

## 🎯 Objectives

- Process healthcare patient data efficiently.
- Use **Apache Kafka** for real-time/event-based data streaming.
- Perform data preprocessing and transformation.
- Build a machine learning pipeline for readmission risk prediction.
- Store and process large-scale healthcare data.
- Provide useful information for discharge planning.
- Demonstrate the integration of Big Data technologies with Machine Learning.

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │   Patient Dataset   │
                 │   diabetic_data.csv │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 │  prepare_data.py    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Apache Kafka     │
                 │  Patient Data Topic │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Processing /   │
                 │ Feature Preparation │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Machine Learning    │
                 │ Classification      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Readmission Risk    │
                 │     Prediction      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Discharge Planning  │
                 │     Dashboard       │
                 └─────────────────────┘
