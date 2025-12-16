from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import joblib

# Load the model from models/
model = joblib.load("models/model.pkl")


app = FastAPI(title="ML Inference API")

# Define input schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    # add more features as per your model

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.feature1, data.feature2]])
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
