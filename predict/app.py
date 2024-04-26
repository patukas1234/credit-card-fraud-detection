import logging
import uvicorn

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader

from models import PredictionRequest, PredictionResponse, FraudPredictor
from config import settings

logger = logging.getLogger(__name__)

api_token_header = APIKeyHeader(name="Auth-Token")
app = FastAPI(title="Fraud Detection API", version="0.0.1")


async def _require_api_token(api_token_header: str = Security(api_token_header)) -> str:
    if api_token_header != settings.api_token:
        raise HTTPException(status_code=401)
    else:
        return api_token_header


async def get_fraud_predictor():
    model_path = settings.model_path
    return FraudPredictor(model_path)


@app.post("/predict")
async def predict(
    request: PredictionRequest,
    api_token: str = Depends(_require_api_token),
    predictor: FraudPredictor = Depends(get_fraud_predictor),
):
    input_features = request.features
    try:
        prediction, prediction_probability = predictor.predict(input_features)
        response = PredictionResponse(
            account_id=request.account_id,
            prediction=prediction[0],
            probability_fraud=prediction_probability[0][1],
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    return response


if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
