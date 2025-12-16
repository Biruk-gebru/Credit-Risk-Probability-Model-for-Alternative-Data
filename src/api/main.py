import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from .pydantic_models import CreditRiskInput, CreditRiskOutput

# Add the root directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import custom transformers so pickle can find them
from src.data_processing import CategoricalEncoder, NumericalScaler, MissingValueImputer

app = FastAPI(title="Credit Risk Prediction API", version="1.0")

MODEL_PATH = "models/model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict", response_model=CreditRiskOutput)
def predict(input_data: CreditRiskInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        data = input_data.dict()
        df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return CreditRiskOutput(prediction=int(prediction), probability=float(probability))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to Credit Risk Prediction API"}
