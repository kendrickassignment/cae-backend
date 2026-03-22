# Corporate Accountability Engine (CAE) — Docker Container
# Deploy on: Google Cloud Run (GCP)

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create upload and results directories
RUN mkdir -p uploads results

# Cloud Run uses PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

# Run the server
# Cloud Run sends SIGTERM for graceful shutdown — uvicorn handles it natively
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
