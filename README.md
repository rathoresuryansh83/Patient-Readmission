# Patient Readmission Prediction Using Kafka, HDFS, Machine Learning, and Dashboard

## Project Overview

This project builds an end-to-end patient readmission prediction pipeline using:

- Python
- Apache Kafka
- Hadoop Distributed File System (HDFS)
- Machine Learning
- Streamlit Dashboard
- Docker

The project uses the **UCI Diabetes 130-US Hospitals dataset** to process patient encounter records, train a machine learning model, generate readmission predictions, and display prediction results in an interactive dashboard.

The project is configured to run on a single Windows PC, with Kafka and HDFS running locally and Docker used to run the dashboard.

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
        Prediction Output
                |
                v
       Streamlit Dashboard
```

## Machine Learning and Prediction

The machine learning pipeline includes preprocessing, Logistic Regression training, test-set evaluation, and readmission probability generation.

### Model Details

- **Algorithm:** Logistic Regression
- **Maximum iterations:** 1,000
- **Class balancing:** Enabled
- **Solver:** `liblinear`
- **Random state:** 42

### Dataset and Evaluation

| Metric | Value |
|---|---:|
| Training rows | 81,412 |
| Testing rows | 20,354 |
| Transformed features | 2,370 |
| Accuracy | 0.6423 |
| Precision | 0.1660 |
| Recall | 0.5482 |
| F1-score | 0.2548 |
| ROC-AUC | 0.6441 |

These metrics are from the existing model evaluation output. Precision is relatively low, so predicted high-risk patients should be treated as candidates for further review rather than as confirmed readmissions.

### Prediction Output

Prediction files are stored locally at:

```text
data/ml/predictions.csv
```

The prediction output has also been verified in HDFS at:

```text
/patient-readmission/predictions/predictions.csv
```

The HDFS prediction file contains 20,354 prediction records plus one header row.

The prediction data includes:

- Patient ID
- Actual 30-day readmission label
- Predicted 30-day readmission label
- Readmission probability

The prediction generation script is located at:

```text
ml/predict.py
```

The trained Logistic Regression model and evaluation metrics are stored under `data/ml/`.

## Interactive Dashboard

The project includes a Streamlit dashboard with a dark navy-and-teal executive design. It displays the generated prediction data and model evaluation results.

### Dashboard Pages

#### 1. Executive Overview

Provides a high-level view of patient readmission risk and key prediction metrics.

#### 2. Patient Risk Explorer

Allows users to explore patient-level predictions and examine risk scores and patient records.

#### 3. Model & Insights

Presents machine learning performance metrics and model-related insights.

### Risk Classification

The current dashboard uses a high-risk threshold of `0.6`.

- **High risk:** Risk score greater than 0.6
- **Low risk:** Risk score less than or equal to 0.6

Risk classifications are based on model-generated probabilities and are intended for analytical exploration, not clinical decision-making.

## Running the Dashboard with Docker

Docker Desktop must be running before starting the dashboard.

From PowerShell, navigate to the project directory:

```powershell
cd C:\Users\surya\patient-readmission
```

Build and start the dashboard:

```powershell
docker compose up --build -d
```

Check the container status:

```powershell
docker compose ps
```

View dashboard logs:

```powershell
docker compose logs --tail 20 dashboard
```

Open the dashboard in your browser:

**http://localhost:8502**

The Docker Compose configuration maps host port `8502` to Streamlit's container port `8501`. Project data directories are mounted into the container so the dashboard can access the available prediction and training data.

To stop the dashboard:

```powershell
docker compose down
```

## Project Structure

```text
patient-readmission/
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   └── data/
├── data/
│   ├── cleaned_data.csv
│   └── ml/
│       ├── predictions.csv
│       ├── logistic_regression.joblib
│       └── model_metrics.txt
├── ml/
│   ├── train.py
│   └── predict.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

This structure reflects the principal files used by the current machine learning and dashboard setup. Additional Kafka and HDFS scripts or configuration files may exist in the repository.

## Technologies Used

- **Python** — data processing and machine learning
- **Apache Kafka** — streaming patient encounter records
- **HDFS** — distributed file storage
- **scikit-learn** — model training and evaluation
- **Pandas** — data manipulation
- **Streamlit** — interactive dashboard
- **Docker and Docker Compose** — dashboard containerization

## Limitations

- The model's performance is moderate, with a ROC-AUC of 0.6441 and precision of 0.1660 on the reported test set.
- The high-risk threshold is configurable and affects how patients are classified.
- Model predictions are not a substitute for professional medical judgment.
- This project is intended for educational and analytical purposes, not direct clinical use.

## Future Improvements

- Evaluate additional classification algorithms.
- Improve precision and overall predictive performance.
- Add model explainability using feature importance or SHAP.
- Expand dashboard filtering and model evaluation.
- Automate prediction refreshes as new data becomes available.
- Improve monitoring and integration across the Kafka, HDFS, and ML pipeline.
