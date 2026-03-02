"""
Data loading utilities for Online Retail Analysis.
"""
import pandas as pd
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Handles loading data from various sources."""
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            db_url: Database connection URL. If None, uses settings.
        """
        self.db_url = db_url or settings.database_url
        self._engine: Optional[Engine] = None
    
    @property
    def engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            logger.info(f"Creating database engine: {self.db_url.split('@')[-1]}")
            self._engine = create_engine(self.db_url)
        return self._engine
    
    def load_excel(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            file_path: Path to Excel file. Defaults to data/Online Retail.xlsx
            
        Returns:
            DataFrame with raw data
        """
        if file_path is None:
            file_path = settings.DATA_DIR / "Online Retail.xlsx"
        
        logger.info(f"Loading data from Excel: {file_path}")
        df = pd.read_excel(file_path, engine="openpyxl")
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with data
        """
        logger.info(f"Loading data from CSV: {file_path}")
        df = pd.read_csv(file_path)
        
        # Parse dates if InvoiceDate column exists
        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def load_from_db(
        self,
        table_name: str,
        query: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data from database.
        
        Args:
            table_name: Name of table to load
            query: Optional SQL query. If None, loads entire table.
            
        Returns:
            DataFrame with database data
        """
        if query is None:
            query = f'SELECT * FROM "{table_name}"'
        
        logger.info(f"Loading data from database table: {table_name}")
        df = pd.read_sql(query, self.engine)
        
        # Parse dates if InvoiceDate column exists
        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def load_clean_sales(self) -> pd.DataFrame:
        """
        Load cleaned sales data.
        
        Returns:
            DataFrame with clean sales data
        """
        # Try database first, then CSV
        try:
            return self.load_from_db("sales_clean")
        except Exception as e:
            logger.warning(f"Could not load from database: {e}")
            csv_path = settings.OUTPUTS_DIR / "online_retail_sales_clean.csv"
            if csv_path.exists():
                return self.load_csv(csv_path)
            raise FileNotFoundError(
                "Clean sales data not found in database or outputs folder"
            )
    
    def save_to_db(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> None:
        """
        Save DataFrame to database.
        
        Args:
            df: DataFrame to save
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
        """
        logger.info(f"Saving {len(df):,} rows to database table: {table_name}")
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        logger.info("Data saved successfully")
