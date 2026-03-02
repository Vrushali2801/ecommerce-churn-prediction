"""
RFM segmentation endpoints.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from typing import List

from api.models import RFMSegmentResponse
from src.data.data_loader import DataLoader
from src.models.rfm_segmentation import RFMSegmentation
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


@router.get(
    "/segments",
    response_model=List[RFMSegmentResponse],
    tags=["rfm"],
    summary="Get RFM segment summary",
)
async def get_rfm_segments():
    """
    Get summary of all RFM customer segments.
    
    Returns statistics for each segment including customer count,
    average recency/frequency/monetary values.
    """
    try:
        # Load data
        loader = DataLoader()
        df = loader.load_clean_sales()
        
        # Run RFM analysis
        rfm_analyzer = RFMSegmentation()
        rfm_df, summary = rfm_analyzer.run_full_analysis(df)
        
        # Convert to response format
        segments = [
            RFMSegmentResponse(
                segment=row["Segment"],
                customer_count=int(row["customer_count"]),
                avg_recency=float(row["avg_recency"]),
                avg_frequency=float(row["avg_frequency"]),
                total_revenue=float(row["total_revenue"]),
                avg_revenue=float(row["avg_revenue"]),
            )
            for _, row in summary.iterrows()
        ]
        
        return segments
    
    except Exception as e:
        logger.error(f"Error getting RFM segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/customers/{customer_id}/segment",
    tags=["rfm"],
    summary="Get segment for specific customer",
)
async def get_customer_segment(customer_id: int):
    """
    Get RFM segment for a specific customer.
    """
    try:
        # Load data
        loader = DataLoader()
        df = loader.load_clean_sales()
        
        # Check if customer exists
        if customer_id not in df["CustomerID"].values:
            raise HTTPException(
                status_code=404,
                detail=f"Customer {customer_id} not found",
            )
        
        # Run RFM analysis
        rfm_analyzer = RFMSegmentation()
        rfm_df, _ = rfm_analyzer.run_full_analysis(df)
        
        # Get customer data
        customer_data = rfm_df[rfm_df["CustomerID"] == customer_id].iloc[0]
        
        return {
            "CustomerID": int(customer_data["CustomerID"]),
            "recency": int(customer_data["recency"]),
            "frequency": int(customer_data["frequency"]),
            "monetary": float(customer_data["monetary"]),
            "R_score": int(customer_data["R_score"]),
            "F_score": int(customer_data["F_score"]),
            "M_score": int(customer_data["M_score"]),
            "RFM_Score": str(customer_data["RFM_Score"]),
            "Segment": str(customer_data["Segment"]),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
