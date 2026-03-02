"""
FastAPI main application.
Online Retail Churn Prediction API.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import health, churn, rfm, customer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Online Retail Churn Prediction API",
    description="""
    API for predicting customer churn and analyzing customer segments.
    
    ## Features
    
    * **Churn Prediction**: Predict which customers are likely to churn
    * **RFM Analysis**: Segment customers based on Recency, Frequency, Monetary values
    * **Batch Processing**: Handle multiple predictions at once
    
    ## Endpoints
    
    * `/api/v1/churn/predict`: Single customer churn prediction
    * `/api/v1/churn/predict-batch`: Batch churn predictions
    * `/api/v1/rfm/segments`: Get all RFM segments
    * `/api/v1/rfm/customers/{id}/segment`: Get specific customer segment
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(churn.router, prefix="/api/v1/churn", tags=["churn"])
app.include_router(rfm.router, prefix="/api/v1/rfm", tags=["rfm"])
app.include_router(customer.router, prefix="/api/v1/customer", tags=["customer"])

# Root redirect
app.include_router(health.router)


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 80)
    logger.info("Starting Online Retail Churn Prediction API")
    logger.info("=" * 80)
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down API")


if __name__ == "__main__":
    import uvicorn
    from src.config import settings
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
