from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn

# ---------------------------
# 1️⃣ Define input schema
# ---------------------------
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ---------------------------
# 2️⃣ Initialize FastAPI app
# ---------------------------
app = FastAPI(title="Iris Classification API")

# ---------------------------
# 3️⃣ Load latest MLflow model
# ---------------------------
def load_latest_model(experiment_name="Iris-Classification"):
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.accuracy DESC"]
        )
        if not runs:
            raise ValueError("No runs found in the experiment.")
        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/iris_model"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load MLflow model: {e}")

model = load_latest_model()

# ---------------------------
# 4️⃣ Define /predict endpoint
# ---------------------------
@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        df = pd.DataFrame([features.dict()])
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# 5️⃣ Healthcheck endpoint
# ---------------------------
@app.get("/")
def root():
    return {"message": "Iris Classification API is up and running!"}

