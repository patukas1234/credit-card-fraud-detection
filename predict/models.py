import joblib

from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    account_id: str
    features: List[float]

class PredictionResponse(BaseModel):
    account_id: str
    prediction: int
    probability_fraud: float

class FraudPredictor:
    def __init__(self, path):
        self.model = joblib.load(path)       
        
    def predict(self, features): 
        predictions = self.model.predict([features])
        prediction_probabilities = self.model.predict_proba([features])
        return predictions, prediction_probabilities
