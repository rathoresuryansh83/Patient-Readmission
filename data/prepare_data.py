import pandas as pd
import os

# ============================================================
# PERSON 1 - PATIENT DATA PREPARATION
# ============================================================

INPUT_FILE = "diabetic_data.csv"

TRAINING_FILE = "training_data.csv"
KAFKA_FILE = "kafka_data.csv"


print("=" * 70)
print("PATIENT READMISSION DATA PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load original dataset
# ------------------------------------------------------------

print("\n[1/7] Loading original dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows    : {len(df)}")
print(f"Original columns : {len(df.columns)}")


# ------------------------------------------------------------
# 2. Convert '?' into missing values
# ------------------------------------------------------------

print("\n[2/7] Handling missing-value markers...")

df = df.replace("?", pd.NA)

print("Question-mark values converted to missing values.")


# ------------------------------------------------------------
# 3. Create 30-day readmission target
# ------------------------------------------------------------

print("\n[3/7] Creating 30-day readmission target...")

df["readmitted_30_days"] = (
    df["readmitted"].astype("string").str.strip() == "<30"
).astype(int)

print("Target definition:")
print("  <30  -> 1 (readmitted within 30 days)")
print("  >30  -> 0")
print("  NO   -> 0")


# ------------------------------------------------------------
# 4. Select features available at discharge
# ------------------------------------------------------------

print("\n[4/7] Selecting discharge-time features...")

selected_features = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]

# Make sure all requested columns exist
missing_columns = [
    column for column in selected_features
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: The following columns are missing:")
    for column in missing_columns:
        print(" -", column)
    raise SystemExit(1)

prepared = df[selected_features + ["readmitted_30_days"]].copy()

print(f"Selected features: {len(selected_features)}")


# ------------------------------------------------------------
# 5. Remove records without essential identifiers
# ------------------------------------------------------------

print("\n[5/7] Validating patient identifiers...")

before = len(prepared)

prepared = prepared.dropna(
    subset=["encounter_id", "patient_nbr"]
)

removed = before - len(prepared)

print(f"Records removed : {removed}")
print(f"Records remaining: {len(prepared)}")


# ------------------------------------------------------------
# 6. Create two datasets
# ------------------------------------------------------------

print("\n[6/7] Creating training and Kafka datasets...")

# Dataset for ML training
training_data = prepared.copy()

# Dataset for Kafka streaming
# IMPORTANT:
# The target is removed because it is the value we want
# the ML model to predict.
kafka_data = prepared.drop(
    columns=["readmitted_30_days"]
).copy()


# ------------------------------------------------------------
# 7. Save files
# ------------------------------------------------------------

print("\n[7/7] Saving files...")

training_data.to_csv(
    TRAINING_FILE,
    index=False
)

kafka_data.to_csv(
    KAFKA_FILE,
    index=False
)

print("\nFiles successfully created:")

print(f"  {TRAINING_FILE}")
print(f"  {KAFKA_FILE}")


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nTraining dataset:")
print(f"  Rows    : {len(training_data)}")
print(f"  Columns : {len(training_data.columns)}")

print("\nKafka dataset:")
print(f"  Rows    : {len(kafka_data)}")
print(f"  Columns : {len(kafka_data.columns)}")

print("\nTarget distribution:")
print(
    training_data["readmitted_30_days"]
    .value_counts()
    .sort_index()
)

print("\nOutput files are ready for the next stages.")