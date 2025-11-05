import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import os

os.makedirs("models", exist_ok=True)

# Load processed data
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train = train.drop("species", axis=1)
y_train = train["species"]
X_test = test.drop("species", axis=1)
y_test = test["species"]

# Start MLflow run
mlflow.set_experiment("Iris-Classification")
with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc}")

    # Log metrics and model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "iris_model")
    joblib.dump(model, "models/iris_model.pkl")

