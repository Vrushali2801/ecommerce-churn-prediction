"""
Churn prediction endpoints.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from api.models import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    BatchChurnPredictionRequest,
    BatchChurnPredictionResponse,
)
from src.models.churn_model import ChurnPredictor
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Global model instance (loaded once)
_model = None


def get_model() -> ChurnPredictor:
    """Dependency to get or load the model."""
    global _model
    if _model is None:
        model_path = settings.MODELS_DIR / "churn_model.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Please train the model first.",
            )
        logger.info(f"Loading model from {model_path}")
        _model = ChurnPredictor(model_path=model_path)
        logger.info("Model loaded successfully")
    return _model


@router.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["churn"],
    summary="Predict churn for a single customer",
)
async def predict_churn(
    request: ChurnPredictionRequest,
    model: ChurnPredictor = Depends(get_model),
):
    """
    Predict churn probability for a single customer.
    
    Returns churn probability (0-1), binary prediction, and risk level.
    """
    try:
        # Convert to DataFrame
        features_dict = request.customer_features.model_dump()
        customer_id = features_dict["CustomerID"]
        
        # Remove CustomerID from features for prediction
        feature_data = {k: v for k, v in features_dict.items() if k != "CustomerID"}
        df = pd.DataFrame([feature_data])
        
        # Add CustomerID back for tracking
        df["CustomerID"] = customer_id
        
        # Get prediction
        result = model.get_customer_risk(df)
        
        return ChurnPredictionResponse(
            CustomerID=int(result.iloc[0]["CustomerID"]),
            churn_probability=float(result.iloc[0]["churn_probability"]),
            churn_prediction=int(result.iloc[0]["churn_prediction"]),
            risk_level=str(result.iloc[0]["risk_level"]),
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict-batch",
    response_model=BatchChurnPredictionResponse,
    tags=["churn"],
    summary="Predict churn for multiple customers",
)
async def predict_churn_batch(
    request: BatchChurnPredictionRequest,
    model: ChurnPredictor = Depends(get_model),
):
    """
    Predict churn probability for multiple customers at once.
    
    More efficient than calling /predict multiple times.
    """
    try:
        # Convert all customers to DataFrame
        customers_data = [c.model_dump() for c in request.customers]
        df = pd.DataFrame(customers_data)
        
        # Separate CustomerID
        customer_ids = df["CustomerID"]
        feature_cols = [col for col in df.columns if col != "CustomerID"]
        
        # Get predictions
        results = model.get_customer_risk(df)
        
        # Convert to response format
        predictions = [
            ChurnPredictionResponse(
                CustomerID=int(row["CustomerID"]),
                churn_probability=float(row["churn_probability"]),
                churn_prediction=int(row["churn_prediction"]),
                risk_level=str(row["risk_level"]),
            )
            for _, row in results.iterrows()
        ]
        
        return BatchChurnPredictionResponse(predictions=predictions)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/risk-distribution",
    tags=["churn"],
    summary="Get risk level distribution",
)
async def get_risk_distribution(model: ChurnPredictor = Depends(get_model)):
    """
    Get feature importance from the trained model.
    """
    try:
        if model.feature_importance_ is None:
            raise HTTPException(
                status_code=404,
                detail="Feature importance not available",
            )
        
        importance = model.feature_importance_.head(10).to_dict(orient="records")
        return {"top_features": importance}
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
