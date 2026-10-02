from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "data" / "ml"

X_TRAIN_FILE = ML_DIR / "X_train.joblib"
X_TEST_FILE = ML_DIR / "X_test.joblib"
Y_TRAIN_FILE = ML_DIR / "y_train.csv"
Y_TEST_FILE = ML_DIR / "y_test.csv"

MODEL_FILE = ML_DIR / "logistic_regression.joblib"
METRICS_FILE = ML_DIR / "model_metrics.txt"


def main():
    print("Loading training and testing data...")

    X_train = joblib.load(X_TRAIN_FILE)
    X_test = joblib.load(X_TEST_FILE)

    y_train = pd.read_csv(Y_TRAIN_FILE)["readmitted_30_days"]
    y_test = pd.read_csv(Y_TEST_FILE)["readmitted_30_days"]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    # Predictions.
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)
    confusion = confusion_matrix(y_test, y_pred)

    print()
    print("MODEL EVALUATION")
    print("----------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print()
    print("Confusion Matrix:")
    print(confusion)
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model.
    joblib.dump(model, MODEL_FILE)

    # Save metrics.
    metrics_text = (
        "Logistic Regression\n"
        "===================\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1 Score : {f1:.4f}\n"
        f"ROC-AUC  : {roc_auc:.4f}\n\n"
        "Confusion Matrix:\n"
        f"{confusion}\n\n"
        "Classification Report:\n"
        f"{classification_report(y_test, y_pred, zero_division=0)}"
    )

    METRICS_FILE.write_text(
        metrics_text,
        encoding="utf-8",
    )

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()