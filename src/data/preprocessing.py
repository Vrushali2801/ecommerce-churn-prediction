"""
Data preprocessing functions.
Refactored from notebooks for production use.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """Handles data cleaning and preprocessing."""
    
    def __init__(self, remove_outliers: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            remove_outliers: Whether to remove statistical outliers
        """
        self.remove_outliers = remove_outliers
    
    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw retail data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")
        df = df.copy()
        
        # Convert data types
        df = self._convert_dtypes(df)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df):,} duplicate rows")
        
        # Flag returns and cancellations
        df = self._flag_returns(df)
        
        # Filter to sales only (no returns/cancellations)
        df = df[~df["is_return_or_cancel"]].copy()
        logger.info(f"Filtered to sales only: {len(df):,} rows")
        
        # Remove rows with missing CustomerID
        df = df.dropna(subset=["CustomerID"])
        logger.info(f"After removing missing CustomerID: {len(df):,} rows")
        
        # Remove invalid prices/quantities
        df = self._remove_invalid_values(df)
        
        # Remove outliers if specified
        if self.remove_outliers:
            df = self._remove_outliers(df)
        
        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        logger.info("Converting data types")
        
        # Invoice date
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        
        # Invoice number as string
        df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
        
        # Customer ID as nullable integer
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
        df["CustomerID"] = df["CustomerID"].astype("Int64")
        
        # Numeric columns
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
        
        # Calculate revenue
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]
        
        return df
    
    def _flag_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag returns and cancellations."""
        df["is_return_qty"] = df["Quantity"] < 0
        df["is_cancel_invoice"] = df["InvoiceNo"].str.startswith("C", na=False)
        df["is_return_or_cancel"] = df["is_return_qty"] | df["is_cancel_invoice"]
        
        n_returns = df["is_return_or_cancel"].sum()
        logger.info(f"Flagged {n_returns:,} returns/cancellations")
        
        return df
    
    def _remove_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid prices or quantities."""
        initial_len = len(df)
        
        # Remove zero/negative prices
        df = df[df["UnitPrice"] > 0]
        
        # Remove zero quantities (already removed negatives)
        df = df[df["Quantity"] > 0]
        
        logger.info(f"Removed {initial_len - len(df):,} invalid value rows")
        return df
    
    def _remove_outliers(
        self,
        df: pd.DataFrame,
        price_quantile: float = 0.999,
        qty_quantile: float = 0.999,
    ) -> pd.DataFrame:
        """Remove statistical outliers."""
        initial_len = len(df)
        
        # Calculate thresholds
        price_threshold = df["UnitPrice"].quantile(price_quantile)
        qty_threshold = df["Quantity"].quantile(qty_quantile)
        
        # Filter
        df = df[
            (df["UnitPrice"] <= price_threshold) &
            (df["Quantity"] <= qty_threshold)
        ]
        
        logger.info(
            f"Removed {initial_len - len(df):,} outlier rows "
            f"(UnitPrice > {price_threshold:.2f} or Quantity > {qty_threshold:.0f})"
        )
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for modeling by ensuring required columns exist.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame ready for modeling
        """
        required_cols = ["InvoiceDate", "CustomerID", "InvoiceNo", "Revenue"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by date
        df = df.sort_values("InvoiceDate").reset_index(drop=True)
        
        return df
