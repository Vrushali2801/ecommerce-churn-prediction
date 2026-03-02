"""
Health check endpoints.
"""
from fastapi import APIRouter
from api.models import HealthResponse
from src.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check API health status."""
    # Try to import the model to check if it's loadable
    model_loaded = False
    try:
        from src.models.churn_model import ChurnPredictor
        model_path = settings.MODELS_DIR / "churn_model.pkl"
        if model_path.exists():
            model_loaded = True
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        mlflow_uri=settings.MLFLOW_TRACKING_URI,
    )


@router.get("/", tags=["health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Online Retail Churn Prediction API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
