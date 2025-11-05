.PHONY: setup dvc-pull dvc-push prepare train run docker-build docker-run clean help

# 1️⃣ Setup environment
setup:
	@echo "🔧 Setting up virtual environment..."
	python3 -m venv venv
	@echo "📦 Installing dependencies..."
	source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✅ Setup complete."

# 2️⃣ DVC pull
dvc-pull:
	@echo "📥 Pulling data from DVC remote..."
	source venv/bin/activate && dvc pull
	@echo "✅ Data pulled."

# 3️⃣ DVC push
dvc-push:
	@echo "📤 Pushing data to DVC remote..."
	source venv/bin/activate && dvc push
	@echo "✅ Data pushed."

# 4️⃣ Prepare data
prepare:
	@echo "🧹 Preparing data..."
	source venv/bin/activate && python src/prepare_data.py
	@echo "✅ Data preparation complete."

# 5️⃣ Train model
train:
	@echo "🚀 Training ML model..."
	source venv/bin/activate && python src/train.py
	@echo "✅ Training complete."

# 6️⃣ Run FastAPI
run:
	@echo "🏃 Running FastAPI..."
	source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 7️⃣ Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t mlops-fasttrack:latest .

# 8️⃣ Run Docker container
docker-run:
	@echo "🚀 Running Docker container..."
	docker run -d -p 8000:8000 mlops-fasttrack:latest

# 9️⃣ Clean project
clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf __pycache__ venv mlruns .dvc data/processed models
	@echo "✅ Clean complete."

# 10️⃣ Help
help:
	@echo "Available commands:"
	@echo "  make setup        - Setup virtual environment & install dependencies"
	@echo "  make dvc-pull     - Pull datasets from DVC remote"
	@echo "  make dvc-push     - Push datasets to DVC remote"
	@echo "  make prepare      - Prepare/clean data"
	@echo "  make train        - Train ML model and log with MLflow"
	@echo "  make run          - Run FastAPI app locally"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make clean        - Remove temporary files"

