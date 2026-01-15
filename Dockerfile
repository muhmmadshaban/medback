FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create ML model directory
RUN mkdir -p ml_model

# Copy ML model
COPY ml_model/ ./ml_model/

# Copy application
COPY main.py .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]