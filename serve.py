from fastapi import FastAPI, Request, HTTPException
import joblib
import pandas as pd
import os

from ml_pipeline.data_processing import ChurnFeatureEngineer
from ml_pipeline.counterfactuals import load_explainer

app = FastAPI(title="SageMaker Churn Inference")

# Load assets at startup so they stay in memory
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model_pipeline.pkl")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "data/X_train_raw.csv")

try:
    model = joblib.load(MODEL_PATH)
    explainer = load_explainer(MODEL_PATH, TRAIN_DATA_PATH)
    print("Model and Explainer loaded successfully.")
except Exception as e:
    print(f"Error loading assets: {e}")
    model = None
    explainer = None

@app.get("/ping")
def ping():
    """SageMaker Health Check."""
    if model is not None and explainer is not None:
        return {"status": "Healthy"}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/invocations")
async def invocations(request: Request):
    """SageMaker Prediction Route."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # SageMaker sends data in the request body
        body = await request.json()
        input_df = pd.DataFrame([body])
        
        THRESHOLD = 0.15 
        probability  = model.predict_proba(input_df)[0][1]
        prediction   = 1 if probability >= THRESHOLD else 0
        risk_level   = "High" if prediction == 1 else "Low"

        interventions = []
        if risk_level == "High" and explainer is not None:
            interventions = explainer.get_interventions(
                input_df, num_cfs=3, proximity_weight=0.5, diversity_weight=1.0
            )

        return {
            "prediction_class": int(prediction),
            "churn_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "interventions": interventions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")