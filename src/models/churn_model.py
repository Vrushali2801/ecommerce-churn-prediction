"""
Churn Prediction Model.
Predicts which customers are likely to churn based on purchase behavior.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class ChurnPredictor:
    """Predicts customer churn using Random Forest."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_smote: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize churn predictor.
        
        Args:
            model_path: Path to saved model. If None, creates new model.
            use_smote: Whether to use SMOTE for handling class imbalance
            random_state: Random seed for reproducibility
        """
        self.model_path = model_path
        self.use_smote = use_smote
        self.random_state = random_state
        
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[list] = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        **model_params,
    ) -> Dict[str, Any]:
        """
        Train churn prediction model.
        
        Args:
            X: Feature matrix
            y: Target variable (1=churned, 0=active)
            test_size: Proportion of data for testing
            **model_params: Additional parameters for RandomForestClassifier
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training churn prediction model")
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        logger.info(f"Churn rate in train: {y_train.mean():.2%}")
        logger.info(f"Churn rate in test: {y_test.mean():.2%}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance with SMOTE
        if self.use_smote and y_train.value_counts().min() > 5:
            logger.info("Applying SMOTE for class balance")
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train_scaled, y_train
            )
            logger.info(
                f"After SMOTE: {len(X_train_balanced):,} samples "
                f"(churn rate: {y_train_balanced.mean():.2%})"
            )
        else:
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train
        
        # Default model parameters if not provided
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": self.random_state,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        default_params.update(model_params)
        
        # Train model
        logger.info("Training Random Forest model")
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
            "accuracy": float(self.model.score(X_test_scaled, y_test)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }
        
        logger.info(f"Model accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        
        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        
        logger.info("\nTop 5 important features:")
        for _, row in self.feature_importance_.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn (binary: 0 or 1).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (0=active, 1=churned)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probability.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of churn probabilities (0 to 1)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save model to disk.
        
        Args:
            path: Save path. If None, uses default path.
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            path = settings.MODELS_DIR / "churn_model.pkl"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance_,
            "use_smote": self.use_smote,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to: {path}")
        
        return path
    
    def load_model(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
        """
        logger.info(f"Loading model from: {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.feature_importance_ = model_data.get("feature_importance")
        self.use_smote = model_data.get("use_smote", True)
        
        logger.info("Model loaded successfully")
    
    def get_customer_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get customer churn risk assessment.
        
        Args:
            X: Feature matrix with CustomerID
            
        Returns:
            DataFrame with CustomerID, churn_probability, churn_prediction, risk_level
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        results = pd.DataFrame({
            "CustomerID": X["CustomerID"] if "CustomerID" in X.columns else range(len(X)),
            "churn_probability": probabilities,
            "churn_prediction": predictions,
        })
        
        # Risk segments (include_lowest=True to handle 0.0 probability)
        results["risk_level"] = pd.cut(
            results["churn_probability"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
        
        return results
