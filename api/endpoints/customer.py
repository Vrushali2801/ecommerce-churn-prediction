"""
Customer lookup and prediction endpoint.
Just provide Customer ID, system does the rest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.data.data_loader import DataLoader
from src.features.feature_engineering import ChurnFeatureEngineer
from src.models.churn_model import ChurnPredictor
from src.models.clv_model import CLVCalculator
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Global model instance
_model = None

def get_model():
    """Get or load the model."""
    global _model
    if _model is None:
        model_path = settings.MODELS_DIR / "churn_model.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Please train the model first.",
            )
        _model = ChurnPredictor(model_path=model_path)
    return _model


class CustomerLookupRequest(BaseModel):
    """Request with just customer ID."""
    customer_id: int


class CustomerPredictionResponse(BaseModel):
    """Simple response with all info."""
    customer_id: int
    exists: bool
    
    # Customer stats
    days_since_last_purchase: int = None
    total_purchases: int = None
    total_spent: float = None
    avg_order_value: float = None
    
    # Prediction
    churn_probability: float = None
    churn_prediction: int = None
    risk_level: str = None
    
    # CLV metrics
    clv_1year: float = None
    clv_segment: str = None
    
    # Action message
    message: str = None


@router.post(
    "/lookup",
    response_model=CustomerPredictionResponse,
    tags=["customer"],
    summary="Predict churn by Customer ID only",
)
async def predict_by_customer_id(request: CustomerLookupRequest):
    """
    Just provide a Customer ID and get instant churn prediction.
    
    The system will:
    1. Look up customer in database
    2. Calculate all features automatically
    3. Return prediction with customer stats
    """
    try:
        customer_id = request.customer_id
        
        # Load data
        loader = DataLoader()
        df = loader.load_clean_sales()
        
        # Check if customer exists
        customer_data = df[df["CustomerID"] == customer_id]
        
        if len(customer_data) == 0:
            return CustomerPredictionResponse(
                customer_id=customer_id,
                exists=False,
                message=f"Customer {customer_id} not found in database"
            )
        
        # Calculate features for all customers (or just this one)
        feature_engineer = ChurnFeatureEngineer()
        features_df = feature_engineer.create_features(df, include_target=False)
        
        # Get this customer's features
        customer_features = features_df[features_df["CustomerID"] == customer_id]
        
        if len(customer_features) == 0:
            return CustomerPredictionResponse(
                customer_id=customer_id,
                exists=False,
                message=f"Could not calculate features for customer {customer_id}"
            )
        
        # Calculate CLV
        clv_calculator = CLVCalculator(time_period_days=365)
        clv_data = clv_calculator.get_customer_clv(df, customer_id)
        
        # Get model and predict
        model = get_model()
        result = model.get_customer_risk(customer_features)
        
        # Get customer stats
        stats = customer_features.iloc[0]
        
        # Prepare response
        risk_level = result.iloc[0]["risk_level"]
        prob = float(result.iloc[0]["churn_probability"])
        
        # Generate message
        if risk_level == "Low":
            message = "✅ Active customer - Keep them engaged with regular communication"
        elif risk_level == "Medium":
            message = "⚠️ At-risk customer - Consider sending a special offer or reaching out"
        else:
            message = "🚨 High churn risk - Take immediate action! Contact them today"
        
        return CustomerPredictionResponse(
            customer_id=customer_id,
            exists=True,
            days_since_last_purchase=int(stats["recency_days"]),
            total_purchases=int(stats["frequency"]),
            total_spent=float(stats["monetary_total"]),
            clv_1year=float(clv_data["projected_clv_1year"]) if clv_data else None,
            clv_segment=clv_data["clv_segment"] if clv_data else None,
            avg_order_value=float(stats["monetary_mean"]),
            churn_probability=prob,
            churn_prediction=int(result.iloc[0]["churn_prediction"]),
            risk_level=risk_level,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error in customer lookup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/random",
    tags=["customer"],
    summary="Get a random customer ID for testing",
)
async def get_random_customer():
    """Get a random customer ID from the database for testing."""
    try:
        loader = DataLoader()
        df = loader.load_clean_sales()
        random_customer = df["CustomerID"].sample(1).iloc[0]
        return {"customer_id": int(random_customer)}
    except Exception as e:
        logger.error(f"Error getting random customer: {e}")
        raise HTTPException(status_code=500, detail=str(e))
