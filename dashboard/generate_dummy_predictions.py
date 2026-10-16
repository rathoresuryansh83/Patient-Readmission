from pathlib import Path

import numpy as np
import pandas as pd

N_PATIENTS = 10_000
HIGH_RISK_THRESHOLD = 0.6  # score > 0.6 -> HIGH, otherwise LOW
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "predictions.csv"

rng = np.random.default_rng(seed=42)

# Right-skewed scores with an average near 0.38
risk_scores = rng.beta(2, 3.3, size=N_PATIENTS).round(2)

predictions = pd.DataFrame(
    {
        "patient_id": [f"P{i:05d}" for i in range(1, N_PATIENTS + 1)],
        "risk_score": risk_scores,
    }
)

predictions["risk_level"] = np.where(
    predictions["risk_score"] > HIGH_RISK_THRESHOLD, "HIGH", "LOW"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(predictions)} rows to {OUTPUT_PATH}")
print(f"High risk : {(predictions['risk_level'] == 'HIGH').sum()}")
print(f"Low risk  : {(predictions['risk_level'] == 'LOW').sum()}")
print(f"Avg risk  : {predictions['risk_score'].mean():.2f}")
print()
print(predictions.head())