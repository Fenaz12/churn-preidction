from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

from data_processing import ChurnFeatureEngineer

router = APIRouter()

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model_pipeline.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class CustomerData(BaseModel):
    age: int
    gender: str
    country: str
    customer_segment: str
    tenure_months: int
    contract_type: str
    monthly_fee: float
    total_revenue: float
    monthly_logins: int
    weekly_active_days: int
    avg_session_time: float
    support_tickets: int
    escalations: int
    avg_resolution_time: float
    csat_score: float
    payment_method: str
    complaint_type: str | None = "No_Complaint" 

@router.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
        
    input_data = pd.DataFrame([customer.model_dump()])
    
    # Predict using the loaded pipeline
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return {
            "prediction_class": int(prediction),
            "churn_probability": round(float(probability), 4),
            "risk_level": "High" if probability > 0.5 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))