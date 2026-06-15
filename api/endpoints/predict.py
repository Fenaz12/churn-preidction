from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import Optional

from ml_pipeline.data_processing import ChurnFeatureEngineer
from ml_pipeline.counterfactuals import load_explainer, CounterfactualExplainer

router = APIRouter()

MODEL_PATH      = os.getenv("MODEL_PATH",      "models/churn_model_pipeline.pkl")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "data/X_train_raw.csv")

# 1. Load Model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 2. Load Explainer (Global Variable)
try:
    explainer = load_explainer(MODEL_PATH, TRAIN_DATA_PATH)
    print("DiCE explainer loaded successfully.")
except Exception as e:
    print(f"Warning: DiCE explainer failed to load: {e}")
    explainer = None

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
    complaint_type: str
    payment_failures: int
    last_login_days_ago: int
    usage_growth_rate: float
    nps_score: int
    email_open_rate: float
    marketing_click_rate: float
    features_used: int
    referral_count: int
    signup_channel: str
    discount_applied: str
    price_increase_last_3m: str
    survey_response: str

@router.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    input_df = pd.DataFrame([customer.model_dump()])

    try:
        THRESHOLD = 0.15 
        probability  = model.predict_proba(input_df)[0][1]
        prediction   = 1 if probability >= THRESHOLD else 0
        risk_level   = "High" if prediction == 1 else "Low"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    interventions = []
    if risk_level == "High" and explainer is not None:
        try:
            interventions = explainer.get_interventions(
                input_df, 
                num_cfs=3,                  
                proximity_weight=1.5,    
                diversity_weight=1.0     
            )
        except Exception as e:
            print(f"DiCE error: {e}")
            interventions = []

    return {
        "prediction_class":  int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level":        risk_level,
        "interventions":     interventions
    }