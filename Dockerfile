# Use a lightweight Python base image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY models/ /app/models/
COPY data/X_train_raw.csv /app/data/

COPY ml_pipeline/ /app/ml_pipeline/
COPY serve.py /app/

EXPOSE 8080

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]