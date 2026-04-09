FROM python:3.10-slim

WORKDIR /app

# Install dependencies required for system + ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install ML dependencies efficiently (CPU version of torch for lighter image)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY . .

# Ensure models directory exists
RUN mkdir -p models

EXPOSE 8000

# Use PORT environment variable for Render compatibility
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
