"""
MLflow training script for churn prediction model.
Tracks experiments, logs metrics, and registers models.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score

from src.config import settings
from src.data.data_loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import ChurnFeatureEngineer
from src.models.churn_model import ChurnPredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_churn_model():
    """Train churn prediction model with MLflow tracking."""
    
    # Set MLflow tracking URI - use local file-based tracking
    mlflow_dir = settings.PROJECT_ROOT / "mlflow" / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlflow_dir}")
    mlflow.set_experiment("churn_prediction")
    
    logger.info("=" * 80)
    logger.info("Starting Churn Model Training with MLflow")
    logger.info("=" * 80)
    
    # Start MLflow run
    with mlflow.start_run(run_name="churn_rf_model"):
        
        # Log parameters
        mlflow.log_param("churn_threshold_days", 90)
        mlflow.log_param("use_smote", True)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("test_size", 0.2)
        
        # 1. Load data
        logger.info("\n1. Loading data...")
        loader = DataLoader()
        
        try:
            df = loader.load_clean_sales()
            logger.info(f"Loaded clean sales: {len(df):,} rows")
        except FileNotFoundError:
            logger.info("Clean sales not found, loading and cleaning raw data...")
            df = loader.load_excel()
            preprocessor = DataPreprocessor()
            df = preprocessor.clean_raw_data(df)
            
            # Save cleaned data
            clean_path = settings.OUTPUTS_DIR / "online_retail_sales_clean.csv"
            df.to_csv(clean_path, index=False)
            logger.info(f"Saved cleaned data to {clean_path}")
        
        mlflow.log_param("data_rows", len(df))
        
        # 2. Feature engineering
        logger.info("\n2. Creating features...")
        feature_engineer = ChurnFeatureEngineer(
            churn_threshold_days=90,
        )
        
        features_df = feature_engineer.create_features(df, include_target=True)
        
        mlflow.log_param("num_customers", len(features_df))
        mlflow.log_param("num_features", len(feature_engineer.get_feature_names()))
        mlflow.log_metric("churn_rate", features_df["is_churned"].mean())
        
        # Save features
        features_path = settings.OUTPUTS_DIR / "churn_features.csv"
        features_df.to_csv(features_path, index=False)
        mlflow.log_artifact(str(features_path))
        logger.info(f"Saved features to {features_path}")
        
        # 3. Prepare training data
        logger.info("\n3. Preparing training data...")
        X = features_df[feature_engineer.get_feature_names()]
        y = features_df["is_churned"]
        
        # 4. Train model
        logger.info("\n4. Training model...")
        predictor = ChurnPredictor(use_smote=True, random_state=42)
        
        metrics = predictor.train(
            X, y,
            test_size=0.2,
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
        )
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("train_size", metrics["train_size"])
        mlflow.log_metric("test_size", metrics["test_size"])
        
        # Log classification report metrics
        for class_label in ["0", "1"]:
            for metric_name in ["precision", "recall", "f1-score"]:
                value = metrics["classification_report"][class_label][metric_name]
                mlflow.log_metric(f"{metric_name}_class_{class_label}", value)
        
        # Cross-validation
        logger.info("\n5. Running cross-validation...")
        cv_scores = cross_val_score(
            predictor.model,
            predictor.scaler.transform(X),
            y,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # 6. Save model
        logger.info("\n6. Saving model...")
        model_path = predictor.save_model()
        
        # Log model with MLflow
        mlflow.sklearn.log_model(
            predictor.model,
            "model",
            signature=mlflow.models.infer_signature(
                predictor.scaler.transform(X),
                predictor.model.predict(predictor.scaler.transform(X))
            ),
        )
        
        # Log feature importance
        if predictor.feature_importance_ is not None:
            importance_path = settings.OUTPUTS_DIR / "feature_importance.csv"
            predictor.feature_importance_.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")
        logger.info("=" * 80)
        
        return predictor, metrics


if __name__ == "__main__":
    train_churn_model()
