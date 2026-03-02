"""
RFM (Recency, Frequency, Monetary) Customer Segmentation.
Refactored from notebook 03 for production use.
"""
import pandas as pd
import numpy as np
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RFMSegmentation:
    """Performs RFM analysis and customer segmentation."""
    
    def __init__(self, reference_date: Optional[pd.Timestamp] = None):
        """
        Initialize RFM analyzer.
        
        Args:
            reference_date: Reference date for recency calculation.
                          If None, uses max date in data.
        """
        self.reference_date = reference_date
    
    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        Args:
            df: Transaction data with CustomerID, InvoiceDate, InvoiceNo, Revenue
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        logger.info("Calculating RFM metrics")
        
        # Set reference date if not provided
        if self.reference_date is None:
            self.reference_date = df["InvoiceDate"].max()
        
        logger.info(f"Reference date: {self.reference_date}")
        
        # Calculate RFM
        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (self.reference_date - x.max()).days,
            "InvoiceNo": "nunique",
            "Revenue": "sum",
        })
        
        rfm.columns = ["recency", "frequency", "monetary"]
        rfm = rfm.reset_index()
        
        logger.info(f"Calculated RFM for {len(rfm):,} customers")
        
        return rfm
    
    def score_rfm(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Score customers on R, F, M scales (1-5).
        
        Args:
            rfm: DataFrame with recency, frequency, monetary columns
            
        Returns:
            DataFrame with added score columns
        """
        logger.info("Scoring RFM metrics")
        
        rfm = rfm.copy()
        
        # Recency score (lower recency = higher score)
        rfm["R_score"] = pd.qcut(
            rfm["recency"], 
            q=5, 
            labels=[5, 4, 3, 2, 1],
            duplicates="drop"
        )
        
        # Frequency score (higher frequency = higher score)
        rfm["F_score"] = pd.qcut(
            rfm["frequency"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        )
        
        # Monetary score (higher monetary = higher score)
        rfm["M_score"] = pd.qcut(
            rfm["monetary"],
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        )
        
        # Combined RFM score
        rfm["RFM_Score"] = (
            rfm["R_score"].astype(str) +
            rfm["F_score"].astype(str) +
            rfm["M_score"].astype(str)
        )
        
        return rfm
    
    def segment_customers(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Assign customers to segments based on RFM scores.
        
        Args:
            rfm: DataFrame with R_score, F_score, M_score columns
            
        Returns:
            DataFrame with Segment column added
        """
        logger.info("Assigning customer segments")
        
        rfm = rfm.copy()
        
        def assign_segment(row):
            """Assign segment based on RFM scores."""
            r = int(row["R_score"])
            f = int(row["F_score"])
            m = int(row["M_score"])
            
            # Champions: High R, F, M
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            
            # Loyal: High R, moderate-high F, M
            if r >= 4 and f >= 3 and m >= 3:
                return "Loyal"
            
            # Potential: Moderate R, F, low M
            if r >= 3 and f >= 3 and m <= 2:
                return "Potential"
            
            # At Risk: Low R, high F
            if r <= 2 and f >= 4:
                return "At Risk"
            
            # Hibernating: Low R, F, M
            if r <= 2 and f <= 2 and m <= 2:
                return "Hibernating"
            
            # New Customers: Recent but low F, M
            if r >= 4 and f <= 2:
                return "New Customers"
            
            # Promising: Moderate R, low F but high M
            if r >= 3 and f <= 2 and m >= 3:
                return "Promising"
            
            # Need Attention: Moderate scores
            if r == 3 and f == 3:
                return "Need Attention"
            
            # Lost: Very low R
            if r == 1:
                return "Lost"
            
            # Others: Everything else
            return "Average Customers"
        
        rfm["Segment"] = rfm.apply(assign_segment, axis=1)
        
        # Log segment distribution
        segment_counts = rfm["Segment"].value_counts()
        logger.info("Segment distribution:")
        for segment, count in segment_counts.items():
            pct = count / len(rfm) * 100
            logger.info(f"  {segment}: {count:,} ({pct:.1f}%)")
        
        return rfm
    
    def get_segment_summary(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each segment.
        
        Args:
            rfm: DataFrame with Segment and RFM metrics
            
        Returns:
            DataFrame with segment statistics
        """
        summary = rfm.groupby("Segment").agg({
            "CustomerID": "count",
            "recency": "mean",
            "frequency": "mean",
            "monetary": ["sum", "mean"],
        }).round(2)
        
        summary.columns = [
            "customer_count",
            "avg_recency",
            "avg_frequency",
            "total_revenue",
            "avg_revenue",
        ]
        
        summary = summary.sort_values("total_revenue", ascending=False)
        
        return summary.reset_index()
    
    def run_full_analysis(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete RFM analysis.
        
        Args:
            df: Transaction data
            
        Returns:
            Tuple of (rfm_with_segments, segment_summary)
        """
        logger.info("Running full RFM analysis")
        
        # Calculate RFM
        rfm = self.calculate_rfm(df)
        
        # Score RFM
        rfm = self.score_rfm(rfm)
        
        # Segment customers
        rfm = self.segment_customers(rfm)
        
        # Get summary
        summary = self.get_segment_summary(rfm)
        
        logger.info("RFM analysis complete")
        
        return rfm, summary
