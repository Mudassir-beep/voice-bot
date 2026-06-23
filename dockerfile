FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create a non-root user
RUN useradd -m -u 1000 reem && chown -R reem:reem /app
USER reem

EXPOSE 10000
EXPOSE 8765

# Use run.py to start both servers
CMD ["python", "run.py"]