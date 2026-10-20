import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DB_PATH = BASE_DIR / "mindcare.db"

# SQLite DB Initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            inputs TEXT NOT NULL,
            predicted_stress TEXT NOT NULL,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load Model Artifacts
model = joblib.load(ARTIFACT_DIR / "stress_model.joblib")
with open(ARTIFACT_DIR / "metadata.json") as f:
    metadata = json.load(f)

app = FastAPI(title="MindCare AI Inference API", version="1.0.0")

class SurveySubmission(BaseModel):
    inputs: dict[str, int]

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "active_model": metadata.get("best_model"),
        "accuracy": metadata.get("test_accuracy"),
        "features_count": len(metadata.get("features", {}))
    }

@app.get("/benchmark")
def get_benchmark():
    return metadata.get("model_comparison", {})

@app.post("/predict")
def predict(submission: SurveySubmission):
    try:
        data = submission.inputs
        ordered_features = list(metadata["features"].keys())
        input_df = pd.DataFrame([[data[feat] for feat in ordered_features]], columns=ordered_features)
        
        pred_index = int(model.predict(input_df)[0])
        confidence = float(model.predict_proba(input_df).max())
        label = metadata["classes"][pred_index]
        timestamp = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (timestamp, inputs, predicted_stress, confidence) VALUES (?, ?, ?, ?)",
            (timestamp, json.dumps(data), label, confidence)
        )
        conn.commit()
        conn.close()

        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "timestamp": timestamp,
            "disclaimer": "Academic ML estimation; not a medical diagnosis."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    total = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rows = cursor.execute(
        "SELECT predicted_stress, COUNT(*) FROM predictions GROUP BY predicted_stress"
    ).fetchall()
    conn.close()
    return {"total": total, "distribution": {r[0]: r[1] for r in rows}}