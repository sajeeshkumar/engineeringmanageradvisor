# Use the official Python image as a base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the working directory in the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the environment variable for FastAPI to listen on all network interfaces
ENV UVICORN_CMD="uvicorn app:app --host 0.0.0.0 --port 8000 --reload"

# Command to run the application using Uvicorn
CMD ["sh", "-c", "$UVICORN_CMD"]
