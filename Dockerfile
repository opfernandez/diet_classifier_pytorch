# Use slim Python base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy project files
COPY . .

# Install uv (dependency manager)
RUN pip install uv
RUN uv sync

# Expose FastAPI default port
EXPOSE 8000

# Command to run API with Uvicorn
CMD ["uv", "run", "--frozen", "--extra", "cpu", "scripts/inference_fastapi.py"]