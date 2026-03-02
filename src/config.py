"""
Configuration management for the Online Retail Analysis project.
Loads environment variables and provides centralized configuration.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Database configuration
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=5433, env="DB_PORT")
    DB_NAME: str = Field(default="online_retail", env="DB_NAME")
    DB_USER: str = Field(default="postgres", env="DB_USER")
    DB_PASSWORD: str = Field(default="", env="DB_PASSWORD")
    
    # MLflow configuration
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000", 
        env="MLFLOW_TRACKING_URI"
    )
    MLFLOW_BACKEND_STORE_URI: str = Field(
        default="sqlite:///mlflow/mlflow.db",
        env="MLFLOW_BACKEND_STORE_URI"
    )
    MLFLOW_ARTIFACT_ROOT: str = Field(
        default="./mlflow/artifacts",
        env="MLFLOW_ARTIFACT_ROOT"
    )
    
    # Model configuration
    MODEL_NAME: str = Field(default="churn_model", env="MODEL_NAME")
    MODEL_STAGE: str = Field(default="production", env="MODEL_STAGE")
    
    # API configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_RELOAD: bool = Field(default=True, env="API_RELOAD")
    
    # Application settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @property
    def database_url(self) -> str:
        """Construct database URL for SQLAlchemy."""
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [
            self.OUTPUTS_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            Path(self.MLFLOW_ARTIFACT_ROOT),
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.create_directories()
