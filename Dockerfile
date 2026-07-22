FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies (needed for compilation/scientific packages if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ app/
COPY frontend/ frontend/
COPY data/ data/
COPY models/ models/
COPY scripts/ scripts/
COPY reviews.db /workspace/reviews.db

EXPOSE 8000

# Set environment variable to make sure logs are not buffered
ENV PYTHONUNBUFFERED=1

# Command to run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
