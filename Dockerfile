# Use Python 3.10 as base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy entire project
COPY . /app

# Expose ports (assuming backend runs on 8000, frontend on 8501 for example)
EXPOSE 8000 8501

# Command to run both backend and frontend using supervisord or concurrently (example uses basic bash)
CMD bash -c "python main.py & python frontend.py & wait"
