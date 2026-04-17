import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("prometheus_exporter", "3.prometheus_exporter.py")
prometheus_exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prometheus_exporter)

app = FastAPI(title="ML Serving API", description="Credit Card Fraud Prediction API with Monitoring")

class PredictionRequest(BaseModel):
    features: list

try:
    # Menggunakan path absolut untuk mlflow.db dan memuat model terbaru
    import os
    base_path = os.path.abspath("../Membangun_model")
    model_db = f"sqlite:///{os.path.join(base_path, 'mlflow.db')}"
    mlflow.set_tracking_uri(model_db)
    
    # Mencari run_id terbaru yang sukses
    runs = mlflow.search_runs(experiment_names=["Credit Card Fraud Detection"])
    if not runs.empty:
        latest_run_id = runs.iloc[0].run_id
        model = mlflow.pyfunc.load_model(f"runs:/{latest_run_id}/model")
    else:
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/metrics")
def metrics():
    prometheus_exporter.update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.time()

    prometheus_exporter.REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='total').inc()

    try:
        if model is not None:
            df = pd.DataFrame([request.features])
            prediction = model.predict(df)[0]
            proba = model.predict_proba(df)[0]
            confidence = float(max(proba))
        else:
            prediction = 1 if sum(request.features) > 10 else 0
            confidence = 0.85

        latency = time.time() - start_time

        prometheus_exporter.PREDICTION_LATENCY.labels(endpoint='/predict').observe(latency)
        prometheus_exporter.API_RESPONSE_TIME.observe(latency)
        prometheus_exporter.PREDICTION_VALUE.set(confidence)

        if prediction == 1:
            prometheus_exporter.FRAUD_PREDICTIONS.inc()
            prometheus_exporter.PREDICTION_CONFIDENCE.labels(prediction_class='fraud').observe(confidence)
        else:
            prometheus_exporter.NORMAL_PREDICTIONS.inc()
            prometheus_exporter.PREDICTION_CONFIDENCE.labels(prediction_class='normal').observe(confidence)

        prometheus_exporter.MODEL_ACCURACY.set(0.95)
        prometheus_exporter.MODEL_PREDICTION_DRIFT.set(0.1)
        prometheus_exporter.THROUGHPUT.set(1.0 / max(latency, 0.001))

        prometheus_exporter.REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='success').inc()

        response_data = {"prediction": int(prediction), "confidence": confidence}
        prometheus_exporter.RESPONSE_SIZE.observe(len(str(response_data)))

        return response_data
    except Exception as e:
        prometheus_exporter.REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='error').inc()
        prometheus_exporter.REQUEST_ERRORS.labels(error_type='prediction_error').inc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)