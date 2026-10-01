import pandas as pd

FILE = "diabetic_data.csv"

print("=" * 60)
print("PATIENT READMISSION DATASET INSPECTION")
print("=" * 60)

# Load dataset
df = pd.read_csv(FILE)

# Basic information
print("\n1. DATASET SIZE")
print("-" * 60)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# Column names
print("\n2. COLUMN NAMES")
print("-" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

# First 5 records
print("\n3. FIRST 5 RECORDS")
print("-" * 60)
print(df.head())

# Target distribution
print("\n4. READMISSION DISTRIBUTION")
print("-" * 60)
print(df["readmitted"].value_counts(dropna=False))

# Missing values
print("\n5. TOP 15 COLUMNS WITH MISSING VALUES")
print("-" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing.head(15))

# Question-mark values
print("\n6. QUESTION-MARK VALUES")
print("-" * 60)

question_marks = (df == "?").sum()
question_marks = question_marks[question_marks > 0].sort_values(
    ascending=False
)

print(question_marks.head(15))

print("\n" + "=" * 60)
print("INSPECTION COMPLETED SUCCESSFULLY")
print("=" * 60)