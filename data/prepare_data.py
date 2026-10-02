import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "diabetic_data.csv"
CLEAN_FILE = BASE_DIR / "data" / "cleaned_data.csv"
KAFKA_FILE = BASE_DIR / "data" / "kafka_data.csv"

print("Loading:", INPUT_FILE)

df = pd.read_csv(INPUT_FILE, na_values=["?"], low_memory=False)

print(f"Raw shape: {df.shape}")

# Create binary target:
# 1 = readmitted within 30 days
# 0 = not readmitted within 30 days
df["readmitted_30_days"] = (df["readmitted"] == "<30").astype(int)

# Create an explicit patient identifier.
df["patient_id"] = df["patient_nbr"].astype("int64")

# Keep the original readmitted column for traceability.
# The ML target is readmitted_30_days.

# Save cleaned dataset with target.
df.to_csv(CLEAN_FILE, index=False)

# Kafka should receive patient/discharge information,
# but the prediction target should not be sent as an input feature.
kafka_df = df.drop(columns=["readmitted", "readmitted_30_days"])

kafka_df.to_csv(KAFKA_FILE, index=False)

print("\nCleaning complete.")
print("Cleaned file:", CLEAN_FILE)
print("Kafka file:", KAFKA_FILE)
print("\nCleaned shape:", df.shape)
print("\nTarget distribution:")
print(df["readmitted_30_days"].value_counts().sort_index())

print("\nMissing values after converting '?' to NaN:")
print(df.isna().sum().loc[lambda x: x > 0].sort_values(ascending=False))

print("\nIdentifier validation:")
print("Duplicate encounter_id:", df["encounter_id"].duplicated().sum())
print("Unique patients:", df["patient_id"].nunique())
