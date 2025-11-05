# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy project files
COPY requirements.txt .
COPY app/ ./app
COPY src/ ./src
COPY models/ ./models
COPY mlruns/ ./mlruns
COPY dvc.yaml ./
COPY .dvc/ ./.dvc

# 4️⃣ Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install mlflow dvc[all] uvicorn scikit-learn pandas

# 5️⃣ Expose port for FastAPI
EXPOSE 8000

# 6️⃣ Set environment variables
ENV MLFLOW_TRACKING_URI=/app/mlruns

# 7️⃣ Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

