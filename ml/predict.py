from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "data" / "ml"
MODEL_FILE = ML_DIR / "logistic_regression.joblib"
X_TEST_FILE = ML_DIR / "X_test.joblib"
Y_TEST_FILE = ML_DIR / "y_test.csv"
PATIENT_IDS_FILE = ML_DIR / "test_patient_ids.csv"
PREDICTIONS_FILE = ML_DIR / "predictions.csv"


def main():
    print("Loading model and test data...")

    model = joblib.load(MODEL_FILE)
    X_test = joblib.load(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE)["readmitted_30_days"]
    patient_ids = pd.read_csv(PATIENT_IDS_FILE)["patient_id"]

    if len(patient_ids) != len(y_test):
        raise ValueError("Patient ID count does not match test target count.")

    print(f"Test feature shape: {X_test.shape}")
    print(f"Test target rows: {len(y_test)}")
    print(f"Patient ID rows: {len(patient_ids)}")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "patient_id": patient_ids.to_numpy(),
        "actual_readmitted_30_days": y_test.to_numpy(),
        "predicted_readmitted_30_days": predictions,
        "readmission_probability": probabilities,
    })

    results.to_csv(PREDICTIONS_FILE, index=False)

    print(f"Predictions generated: {len(results)}")
    print(f"Positive predictions: {int(predictions.sum())}")
    print(f"Average predicted probability: {probabilities.mean():.4f}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()