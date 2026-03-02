"""
Pydantic models for API request/response validation.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Features for a single customer prediction."""
    CustomerID: int
    recency_days: float = Field(..., description="Days since last purchase")
    frequency: float = Field(..., description="Number of purchases")
    monetary_total: float = Field(..., description="Total revenue")
    monetary_mean: float = Field(..., description="Average order value")
    monetary_std: float = Field(0.0, description="Std dev of order values")
    total_items: float = Field(..., description="Total items purchased")
    avg_items_per_order: float = Field(..., description="Average items per order")
    max_items_per_order: float = Field(..., description="Max items in one order")
    avg_unit_price: float = Field(..., description="Average unit price")
    max_unit_price: float = Field(..., description="Maximum unit price")
    min_unit_price: float = Field(..., description="Minimum unit price")
    price_range: float = Field(..., description="Price range")
    tenure_days: float = Field(..., description="Days since first purchase")
    active_days: float = Field(..., description="Days between first and last purchase")
    purchase_velocity: float = Field(..., description="Purchases per month")
    avg_days_between_purchases: float = Field(..., description="Avg days between purchases")


class ChurnPredictionRequest(BaseModel):
    """Request model for churn prediction."""
    customer_features: CustomerFeatures


class BatchChurnPredictionRequest(BaseModel):
    """Request model for batch churn predictions."""
    customers: List[CustomerFeatures]


class ChurnPredictionResponse(BaseModel):
    """Response model for churn prediction."""
    CustomerID: int
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: int = Field(..., description="Binary prediction (0=active, 1=churned)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")


class BatchChurnPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[ChurnPredictionResponse]


class CustomerIDRequest(BaseModel):
    """Request model for customer ID lookup."""
    customer_id: int


class RFMSegmentResponse(BaseModel):
    """Response model for RFM segment information."""
    segment: str
    customer_count: int
    avg_recency: float
    avg_frequency: float
    total_revenue: float
    avg_revenue: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    mlflow_uri: Optional[str] = None
