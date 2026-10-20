import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# 1. Load Dataset
df = pd.read_csv(BASE_DIR / "StressLevelDataset.csv")

X = df.drop(columns=['stress_level'])
y = df['stress_level']

# 2. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Model Comparison: Random Forest, SVM, Logistic Regression
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

comparison = {}
best_model_name = None
best_accuracy = 0.0
best_model_obj = None

print("=" * 60)
print(f"Training on 1,100 records across 20 parameters...")
print("=" * 60)

for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")
    
    comparison[name] = {
        "Accuracy": round(float(acc) * 100, 2),
        "Precision": round(float(prec) * 100, 2),
        "Recall": round(float(rec) * 100, 2),
        "F1_Score": round(float(f1) * 100, 2)
    }
    
    print(f"{name:25} | Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model_obj = clf

print("=" * 60)
print(f"Best Performing Model: {best_model_name} ({best_accuracy*100:.2f}%)")

# 4. Save Best Model and Metadata
joblib.dump(best_model_obj, ARTIFACT_DIR / "stress_model.joblib")

feature_metadata = {}
for col in X.columns:
    feature_metadata[col] = {
        "min": int(X[col].min()),
        "max": int(X[col].max()),
        "default": int(X[col].median()),
        "label": col.replace("_", " ").title()
    }

metadata = {
    "features": feature_metadata,
    "classes": ["Low", "Moderate", "High"],
    "best_model": best_model_name,
    "test_accuracy": round(best_accuracy * 100, 2),
    "model_comparison": comparison,
    "total_records": int(len(df))
}

with open(ARTIFACT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("[SUCCESS] Best model and comparison metrics saved in artifacts/")