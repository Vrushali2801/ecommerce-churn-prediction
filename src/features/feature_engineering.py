"""
Feature engineering for churn prediction model.
Creates behavioral and transactional features from customer purchase history.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime, timedelta

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChurnFeatureEngineer:
    """Creates features for churn prediction from transaction data."""
    
    def __init__(
        self,
        reference_date: Optional[pd.Timestamp] = None,
        churn_threshold_days: int = 90,
    ):
        """
        Initialize feature engineer.
        
        Args:
            reference_date: Reference date for calculating features. 
                          If None, uses max date in data.
            churn_threshold_days: Number of days of inactivity to define churn
        """
        self.reference_date = reference_date
        self.churn_threshold_days = churn_threshold_days
    
    def create_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Create all features for churn prediction.
        
        Args:
            df: Transaction data with CustomerID, InvoiceDate, InvoiceNo, Revenue
            include_target: Whether to include the churn target variable
            
        Returns:
            DataFrame with customer-level features (one row per customer)
        """
        logger.info("Creating churn prediction features")
        
        # Set reference date if not provided
        if self.reference_date is None:
            self.reference_date = df["InvoiceDate"].max()
        
        logger.info(f"Reference date: {self.reference_date}")
        
        # Calculate RFM features
        rfm_features = self._calculate_rfm(df)
        
        # Calculate behavioral features
        behavior_features = self._calculate_behavioral_features(df)
        
        # Calculate temporal features
        temporal_features = self._calculate_temporal_features(df)
        
        # Merge all features
        features = rfm_features.merge(
            behavior_features, on="CustomerID", how="left"
        ).merge(
            temporal_features, on="CustomerID", how="left"
        )
        
        # Create target variable if requested
        if include_target:
            features = self._create_target(features, df)
        
        logger.info(
            f"Created {len(features.columns)} features for {len(features):,} customers"
        )
        
        return features
    
    def _calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) features."""
        logger.info("Calculating RFM features")
        
        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (self.reference_date - x.max()).days,
            "InvoiceNo": "nunique",
            "Revenue": ["sum", "mean", "std"],
        })
        
        rfm.columns = [
            "recency_days",
            "frequency",
            "monetary_total",
            "monetary_mean",
            "monetary_std",
        ]
        
        # Fill NaN std with 0 (customers with only 1 purchase)
        rfm["monetary_std"] = rfm["monetary_std"].fillna(0)
        
        rfm = rfm.reset_index()
        return rfm
    
    def _calculate_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate behavioral purchase features."""
        logger.info("Calculating behavioral features")
        
        behavior = df.groupby("CustomerID").agg({
            "Quantity": ["sum", "mean", "max"],
            "UnitPrice": ["mean", "max", "min"],
        })
        
        behavior.columns = [
            "total_items",
            "avg_items_per_order",
            "max_items_per_order",
            "avg_unit_price",
            "max_unit_price",
            "min_unit_price",
        ]
        
        behavior = behavior.reset_index()
        
        # Price range
        behavior["price_range"] = (
            behavior["max_unit_price"] - behavior["min_unit_price"]
        )
        
        return behavior
    
    def _calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features."""
        logger.info("Calculating temporal features")
        
        # Customer tenure and purchase intervals
        temporal_agg = df.groupby("CustomerID")["InvoiceDate"].agg([
            ("first_purchase", "min"),
            ("last_purchase", "max"),
            ("num_purchases", "count"),
        ]).reset_index()
        
        # Tenure in days
        temporal_agg["tenure_days"] = (
            self.reference_date - temporal_agg["first_purchase"]
        ).dt.days
        
        # Days since first to last purchase
        temporal_agg["active_days"] = (
            temporal_agg["last_purchase"] - temporal_agg["first_purchase"]
        ).dt.days
        
        # Purchase velocity (purchases per month)
        temporal_agg["purchase_velocity"] = (
            temporal_agg["num_purchases"] / 
            (temporal_agg["tenure_days"] / 30 + 1)  # +1 to avoid division by zero
        )
        
        # Average days between purchases
        temporal_agg["avg_days_between_purchases"] = (
            temporal_agg["active_days"] / 
            (temporal_agg["num_purchases"] + 1)  # +1 to avoid division by zero
        )
        
        # Drop intermediate columns
        temporal = temporal_agg[[
            "CustomerID",
            "tenure_days",
            "active_days",
            "purchase_velocity",
            "avg_days_between_purchases",
        ]]
        
        return temporal
    
    def _create_target(
        self,
        features: pd.DataFrame,
        transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create churn target variable.
        
        A customer is considered churned if they haven't made a purchase 
        in the last churn_threshold_days.
        
        Args:
            features: Feature DataFrame
            transactions: Transaction DataFrame
            
        Returns:
            Features with 'is_churned' column added
        """
        logger.info(
            f"Creating churn target (threshold: {self.churn_threshold_days} days)"
        )
        
        # For each customer, check if recency > threshold
        features["is_churned"] = (
            features["recency_days"] > self.churn_threshold_days
        ).astype(int)
        
        churn_rate = features["is_churned"].mean()
        logger.info(
            f"Churn rate: {churn_rate:.2%} "
            f"({features['is_churned'].sum():,} churned customers)"
        )
        
        return features
    
    def get_feature_names(self) -> list:
        """Get list of feature names (excluding target and CustomerID)."""
        return [
            "recency_days",
            "frequency",
            "monetary_total",
            "monetary_mean",
            "monetary_std",
            "total_items",
            "avg_items_per_order",
            "max_items_per_order",
            "avg_unit_price",
            "max_unit_price",
            "min_unit_price",
            "price_range",
            "tenure_days",
            "active_days",
            "purchase_velocity",
            "avg_days_between_purchases",
        ]
