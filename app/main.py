from fastapi import FastAPI
import mlflow.sklearn
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="ML Model API")

class InputData(BaseModel):
    features: list

# Replace <your_run_id> after training with actual MLflow run ID
model = mlflow.sklearn.load_model("runs:/<your_run_id>/model")

@app.get("/")
def root():
    return {"message": "ML API running"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict(np.array(data.features).reshape(1, -1))
    return {"prediction": int(prediction[0])}
