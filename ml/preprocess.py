from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "cleaned_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "ml"

TARGET = "readmitted_30_days"

# These columns identify an encounter/patient and should not be model features.
ID_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "patient_id",
]

# Original readmission outcome must be excluded to prevent target leakage.
LEAKAGE_COLUMNS = [
    "readmitted",
    TARGET,
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print(f"Dataset shape: {df.shape}")

    # Separate target from model inputs.
    y = df[TARGET].astype("int64")

    columns_to_drop = ID_COLUMNS + LEAKAGE_COLUMNS
    X = df.drop(columns=columns_to_drop)

    print(f"Feature columns: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    # Identify feature types.
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["str", "object"]
    ).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Numeric preprocessing.
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    # Categorical preprocessing.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # Split before fitting the preprocessing pipeline.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Fit preprocessing only on training data.
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Transformed training shape: {X_train_transformed.shape}")
    print(f"Transformed testing shape: {X_test_transformed.shape}")

    # Save transformed feature matrices.
    joblib.dump(
        X_train_transformed,
        OUTPUT_DIR / "X_train.joblib",
    )

    joblib.dump(
        X_test_transformed,
        OUTPUT_DIR / "X_test.joblib",
    )

    # Save targets.
    y_train.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False,
    )

    # Save fitted preprocessing pipeline.
    joblib.dump(
        preprocessor,
        OUTPUT_DIR / "preprocessor.joblib",
    )

    print(f"Saved ML artifacts to: {OUTPUT_DIR}")
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()