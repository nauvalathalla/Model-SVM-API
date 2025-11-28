"""
OCSVM Anomaly Detection FastAPI app for maggot monitoring

Features:
- Training function that creates dummy "normal" data and fits One-Class SVM
- FastAPI endpoints:
  - POST /sensor  -> ingest per-minute sensor reading
  - GET  /status/latest -> get last analysis result
  - POST /train -> retrain model (optional)
- Scheduler job (APScheduler) runs every 3 hours to analyze the last 3 hours of data
- SQLite (SQLAlchemy) stores raw readings and analysis results
- CLI helpers: --train to generate & save model, --simulate to post fake data to a running server

Requirements (pip):
fastapi, uvicorn, sqlalchemy, apscheduler, scikit-learn, numpy, joblib, requests, pydantic

Run instructions (example):
1) Train model:   python ocsvm_anomaly_fastapi.py --train
2) Run server:    python ocsvm_anomaly_fastapi.py
   (server starts on http://0.0.0.0:8000)
3) Simulate data (in another terminal):
   python ocsvm_anomaly_fastapi.py --simulate 180  # posts 180 samples to localhost

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
from sklearn.svm import OneClassSVM
import joblib
import os
import argparse
import time
import requests

# ---------------- CONFIG (tunable) ----------------
MODEL_PATH = "ocsvm_maggot.pkl"
DB_URL = "sqlite:///./sensor_data.db"
WINDOW_HOURS = 3
SAMPLES_PER_MIN = 1  # sensor sends 1 sample per minute

# thresholds for classification based on anomaly rate
RATE_BAHAY = 0.15   # >= 15% anomalies -> BAHAY
RATE_WASPADA = 0.05 # 5% - 14% -> WASPADA

# OCSVM hyperparams (can be tuned)
OCSVM_NU = 0.05
OCSVM_KERNEL = 'rbf'
OCSVM_GAMMA = 'auto'

# --------------------------------------------------
app = FastAPI(title="OCSVM Maggot Anomaly Detector")

# ----- Database setup -----
engine = sa.create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    temperature = sa.Column(sa.Float, nullable=False)
    humidity = sa.Column(sa.Float, nullable=False)
    timestamp = sa.Column(sa.DateTime, default=datetime.utcnow, index=True)

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    start_time = sa.Column(sa.DateTime, nullable=False)
    end_time = sa.Column(sa.DateTime, nullable=False)
    anomaly_count = sa.Column(sa.Integer, nullable=False)
    total_count = sa.Column(sa.Integer, nullable=False)
    rate = sa.Column(sa.Float, nullable=False)
    status = sa.Column(sa.String, nullable=False)  # AMAN / WASPADA / BAHAY
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ----- Pydantic payload -----
class SensorPayload(BaseModel):
    temperature: float
    humidity: float
    timestamp: Optional[datetime] = None

# ----- Global model holder -----
MODEL = None

# ---------------- Utility: train dummy model ----------------
def train_and_save_model(save_path: str = MODEL_PATH, n_samples: int = 1000):
    """Create dummy normal data and train One-Class SVM."""
    # normal ranges
    temps = np.random.uniform(30.0, 36.0, n_samples)
    hums = np.random.uniform(60.0, 80.0, n_samples)
    X_train = np.column_stack([temps, hums])

    model = OneClassSVM(kernel=OCSVM_KERNEL, gamma=OCSVM_GAMMA, nu=OCSVM_NU)
    model.fit(X_train)
    joblib.dump(model, save_path)
    print(f"Model trained and saved to {save_path}")
    return model

# ---------------- Load model ----------------
def load_model(path: str = MODEL_PATH):
    global MODEL
    if os.path.exists(path):
        MODEL = joblib.load(path)
        print(f"Loaded model from {path}")
    else:
        MODEL = None
        print("No model found. Please train first (use --train or POST /train).")

# ---------------- Analysis logic ----------------

def analyze_window(readings: List[SensorReading], model) -> dict:
    """Given ordered readings and a trained OCSVM model, return analysis result."""
    if model is None:
        raise RuntimeError("Model not loaded")

    if not readings:
        return {
            "anomaly_count": 0,
            "total_count": 0,
            "rate": 0.0,
            "status": "AMAN"
        }

    X = np.array([[r.temperature, r.humidity] for r in readings])
    preds = model.predict(X)  # 1 normal, -1 anomaly
    anomaly_count = int(np.sum(preds == -1))
    total = len(preds)
    rate = anomaly_count / total if total > 0 else 0.0

    if rate >= RATE_BAHAY:
        status = "BAHAY"
    elif rate >= RATE_WASPADA:
        status = "WASPADA"
    else:
        status = "AMAN"

    return {
        "anomaly_count": anomaly_count,
        "total_count": total,
        "rate": rate,
        "status": status
    }

# ---------------- Scheduler job ----------------

def job_check_last_window():
    db = SessionLocal()
    end = datetime.utcnow()
    start = end - timedelta(hours=WINDOW_HOURS)

    readings = db.query(SensorReading).filter(
        SensorReading.timestamp >= start, SensorReading.timestamp <= end
    ).order_by(SensorReading.timestamp).all()

    try:
        analysis = analyze_window(readings, MODEL)
    except Exception as e:
        print("Analysis aborted:", e)
        db.close()
        return

    res = AnalysisResult(
        start_time=start,
        end_time=end,
        anomaly_count=analysis['anomaly_count'],
        total_count=analysis['total_count'],
        rate=analysis['rate'],
        status=analysis['status']
    )
    db.add(res)
    db.commit()
    db.close()

    if analysis['status'] == 'BAHAY':
        print(f"[ALERT] BAHAY detected {analysis['anomaly_count']}/{analysis['total_count']} ({analysis['rate']:.2%}) in window {start} - {end}")
    else:
        print(f"Analysis done: status={analysis['status']} anomaly_count={analysis['anomaly_count']} rate={analysis['rate']:.2%}")

# start scheduler
scheduler = BackgroundScheduler()
# run every 3 hours; next_run_time set to utcnow so first run occurs on startup
scheduler.add_job(job_check_last_window, 'interval', hours=3, next_run_time=datetime.utcnow())

# ---------------- FastAPI endpoints ----------------
@app.on_event("startup")
def startup_event():
    load_model()
    scheduler.start()

@app.post("/sensor", status_code=201)
def ingest_sensor(payload: SensorPayload):
    db = SessionLocal()
    ts = payload.timestamp or datetime.utcnow()
    r = SensorReading(temperature=payload.temperature, humidity=payload.humidity, timestamp=ts)
    db.add(r)
    db.commit()
    db.close()
    return {"message": "OK", "timestamp": ts.isoformat()}

@app.get("/status/latest")
def get_latest_status():
    db = SessionLocal()
    res = db.query(AnalysisResult).order_by(AnalysisResult.created_at.desc()).first()
    db.close()
    if not res:
        return {"message": "No analysis yet"}
    return {
        "start_time": res.start_time.isoformat(),
        "end_time": res.end_time.isoformat(),
        "anomaly_count": res.anomaly_count,
        "total_count": res.total_count,
        "rate": res.rate,
        "status": res.status,
        "checked_at": res.created_at.isoformat()
    }

@app.post("/train")
def train_endpoint(samples: int = 1000):
    """Retrain model with dummy normal data. Returns model path."""
    model = train_and_save_model(n_samples=samples)
    load_model()
    return {"message": "Model retrained", "model_path": MODEL_PATH}

# ---------------- Simulation helper (posts fake sensor data) ----------------

def simulate_post(server_url: str = "http://127.0.0.1:8000", n_samples: int = 180, speedup: float = 0.05):
    """Simulate n_samples posts to POST /sensor.
    speedup controls delay between posts in seconds (use <60 to speed test).
    """
    url = server_url.rstrip('/') + "/sensor"
    temp_base = 33.0
    for i in range(n_samples):
        # create normal baseline with occasional spike window
        if 120 <= i <= 140:
            temp = temp_base + (i-120) * 0.12 + np.random.uniform(-0.05, 0.05)  # rising spike
        else:
            temp = temp_base + np.random.uniform(-0.25, 0.25)
        hum = 72.0 + np.random.uniform(-3.0, 3.0)
        payload = {"temperature": float(temp), "humidity": float(hum), "timestamp": datetime.utcnow().isoformat()}
        try:
            resp = requests.post(url, json=payload, timeout=3)
            if resp.status_code != 201:
                print("POST error", resp.status_code, resp.text)
        except Exception as e:
            print("Exception posting data:", e)
        time.sleep(speedup)
    print("Simulation finished")

# ---------------- CLI entrypoint ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCSVM Anomaly Detector server / helper")
    parser.add_argument('--train', action='store_true', help='Train model and save to disk')
    parser.add_argument('--simulate', type=int, default=0, help='Simulate N samples and post to local server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for server')
    parser.add_argument('--port', type=int, default=8000, help='Port for server')
    args = parser.parse_args()

    if args.train:
        train_and_save_model()
        load_model()
    if args.simulate > 0:
        simulate_post(n_samples=args.simulate)

    if not args.simulate:
        # start uvicorn server programmatically
        import uvicorn
        print("Starting server...")
        uvicorn.run("ocsvm_anomaly_fastapi:app", host=args.host, port=args.port, reload=False)
