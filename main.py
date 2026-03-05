import os
import boto3
from fastapi import FastAPI
from dotenv import load_dotenv
from api.endpoints.predict import router as predict_router

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model_pipeline.pkl")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Download from S3 BEFORE doing anything else
if not os.path.exists(MODEL_PATH) and BUCKET_NAME:
    print("Model not found locally. Downloading from S3...")
    os.makedirs('models', exist_ok=True)
    s3 = boto3.client('s3')
    try:
        s3.download_file(BUCKET_NAME, "churn_model_pipeline.pkl", MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download from S3: {e}")


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

app.include_router(predict_router, prefix="", tags=["Predictions"])

