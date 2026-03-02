"""
Customer Lifetime Value (CLV) Calculation.
Simple CLV model based on historical purchase patterns.
"""
import pandas as pd
import numpy as np
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CLVCalculator:
    """Calculate Customer Lifetime Value using historical method."""
    
    def __init__(self, time_period_days: int = 365):
        """
        Initialize CLV calculator.
        
        Args:
            time_period_days: Time period to project CLV (default: 365 days = 1 year)
        """
        self.time_period_days = time_period_days
    
    def calculate_clv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CLV for each customer.
        
        Formula: CLV = (Average Order Value × Purchase Frequency) × Customer Lifespan
        
        Args:
            df: Transaction data with CustomerID, InvoiceDate, Revenue
            
        Returns:
            DataFrame with CLV metrics per customer
        """
        logger.info("Calculating Customer Lifetime Value")
        
        # Calculate metrics per customer
        clv_data = df.groupby("CustomerID").agg({
            "Revenue": ["sum", "mean", "count"],
            "InvoiceNo": "nunique",
            "InvoiceDate": ["min", "max"],
        })
        
        clv_data.columns = ["_".join(col).strip() for col in clv_data.columns.values]
        clv_data = clv_data.rename(columns={
            "Revenue_sum": "total_revenue",
            "Revenue_mean": "avg_transaction_value",
            "Revenue_count": "total_transactions",
            "InvoiceNo_nunique": "total_orders",
            "InvoiceDate_min": "first_purchase",
            "InvoiceDate_max": "last_purchase",
        })
        
        # Calculate customer lifespan in days
        clv_data["lifespan_days"] = (
            clv_data["last_purchase"] - clv_data["first_purchase"]
        ).dt.days
        
        # Avoid division by zero
        clv_data["lifespan_days"] = clv_data["lifespan_days"].replace(0, 1)
        
        # Calculate purchase frequency (orders per day)
        clv_data["purchase_frequency_per_day"] = (
            clv_data["total_orders"] / clv_data["lifespan_days"]
        )
        
        # Calculate average order value
        clv_data["avg_order_value"] = (
            clv_data["total_revenue"] / clv_data["total_orders"]
        )
        
        # Calculate projected CLV for time period
        # CLV = Average Order Value × Purchase Frequency × Time Period
        clv_data["clv_projected"] = (
            clv_data["avg_order_value"] 
            * clv_data["purchase_frequency_per_day"] 
            * self.time_period_days
        )
        
        # Calculate historical CLV (actual revenue)
        clv_data["clv_historical"] = clv_data["total_revenue"]
        
        # Add customer value segment
        clv_data["clv_segment"] = pd.qcut(
            clv_data["clv_projected"],
            q=4,
            labels=["Low Value", "Medium Value", "High Value", "VIP"],
            duplicates="drop",
        )
        
        logger.info(f"Calculated CLV for {len(clv_data):,} customers")
        logger.info(f"Average CLV (1-year): ${clv_data['clv_projected'].mean():,.2f}")
        logger.info(f"Median CLV (1-year): ${clv_data['clv_projected'].median():,.2f}")
        
        return clv_data.reset_index()
    
    def get_customer_clv(self, df: pd.DataFrame, customer_id: int) -> Optional[dict]:
        """
        Get CLV metrics for a specific customer.
        
        Args:
            df: Transaction data
            customer_id: Customer ID to lookup
            
        Returns:
            Dictionary with CLV metrics or None if customer not found
        """
        clv_df = self.calculate_clv(df)
        customer_data = clv_df[clv_df["CustomerID"] == customer_id]
        
        if len(customer_data) == 0:
            return None
        
        row = customer_data.iloc[0]
        return {
            "customer_id": int(row["CustomerID"]),
            "historical_clv": float(row["clv_historical"]),
            "projected_clv_1year": float(row["clv_projected"]),
            "avg_order_value": float(row["avg_order_value"]),
            "purchase_frequency_per_day": float(row["purchase_frequency_per_day"]),
            "lifespan_days": int(row["lifespan_days"]),
            "total_orders": int(row["total_orders"]),
            "clv_segment": str(row["clv_segment"]),
        }
