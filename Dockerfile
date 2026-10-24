FROM python:3.10-slim

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Code + params.yaml + model/model.h5 (see .dockerignore for what is skipped)
COPY . .

EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]
