import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib, json
from pathlib import Path

np.random.seed(42)
N = 1200

# 1. Generate 1200 student records across 12 clear, realistic parameters (Scale 0 to 5)
# 0 = Best condition / No issue, 5 = Worst condition / Severe issue
anxiety = np.random.randint(0, 6, N)
depression = np.random.randint(0, 6, N)
headache_frequency = np.random.randint(0, 6, N)
sleep_deprivation = np.random.randint(0, 6, N)          # 5 = severe insomnia
academic_workload = np.random.randint(0, 6, N)          # 5 = overwhelming load
exam_panic = np.random.randint(0, 6, N)
poor_living_condition = np.random.randint(0, 6, N)      # 5 = very bad environment
lack_of_social_support = np.random.randint(0, 6, N)     # 5 = totally isolated
peer_competition_pressure = np.random.randint(0, 6, N)
future_career_worry = np.random.randint(0, 6, N)
financial_hardship = np.random.randint(0, 6, N)
physical_exhaustion = np.random.randint(0, 6, N)

# Calculate a cumulative stress score (Range: 0 to 60)
raw_score = (
    anxiety * 1.5 +
    depression * 1.5 +
    sleep_deprivation * 1.3 +
    academic_workload * 1.2 +
    exam_panic * 1.1 +
    lack_of_social_support * 1.0 +
    headache_frequency * 0.9 +
    physical_exhaustion * 0.9 +
    future_career_worry * 0.8 +
    peer_competition_pressure * 0.7 +
    financial_hardship * 0.7 +
    poor_living_condition * 0.6
)

# 0: Low Stress (< 25), 1: Moderate Stress (25-45), 2: High Stress (> 45)
stress_level = []
for s in raw_score:
    if s < 25:
        stress_level.append(0)
    elif s < 45:
        stress_level.append(1)
    else:
        stress_level.append(2)

df = pd.DataFrame({
    'anxiety': anxiety,
    'depression': depression,
    'headache_frequency': headache_frequency,
    'sleep_deprivation': sleep_deprivation,
    'academic_workload': academic_workload,
    'exam_panic': exam_panic,
    'poor_living_condition': poor_living_condition,
    'lack_of_social_support': lack_of_social_support,
    'peer_competition_pressure': peer_competition_pressure,
    'future_career_worry': future_career_worry,
    'financial_hardship': financial_hardship,
    'physical_exhaustion': physical_exhaustion,
    'stress_level': stress_level
})

df.to_csv("StressLevelDataset.csv", index=False)
print("1. 'StressLevelDataset.csv' created with consistent directional factors.")

# 2. Train and save model artifacts
X = df.drop(columns=['stress_level'])
y = df['stress_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"2. Model Trained with Test Accuracy: {acc*100:.2f}%")

artifacts = Path("artifacts")
artifacts.mkdir(exist_ok=True)
joblib.dump(model, artifacts / "stress_model.joblib")

features = {}
for col in X.columns:
    features[col] = {
        "min": 0,
        "max": 5,
        "default": 1,
        "label": col.replace("_", " ").title()
    }

meta = {
    "features": features,
    "classes": ["Low", "Moderate", "High"],
    "test_accuracy": round(acc * 100, 2),
    "best_model": "Random Forest"
}

with open(artifacts / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("3. Saved artifacts/stress_model.joblib and artifacts/metadata.json.")