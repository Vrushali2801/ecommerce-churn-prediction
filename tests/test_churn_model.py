"""
Tests for churn model.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.churn_model import ChurnPredictor
from src.features.feature_engineering import ChurnFeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n_customers = 100
    
    data = []
    for customer_id in range(1, n_customers + 1):
        n_transactions = np.random.randint(1, 20)
        for _ in range(n_transactions):
            data.append({
                'CustomerID': customer_id,
                'InvoiceDate': pd.Timestamp('2023-01-01') + 
                              pd.Timedelta(days=np.random.randint(0, 365)),
                'InvoiceNo': f'INV{np.random.randint(1000, 9999)}',
                'Revenue': np.random.uniform(10, 500),
                'Quantity': np.random.randint(1, 10),
                'UnitPrice': np.random.uniform(5, 50),
            })
    
    return pd.DataFrame(data)


def test_feature_engineering(sample_data):
    """Test feature engineering process."""
    engineer = ChurnFeatureEngineer(churn_threshold_days=90)
    features = engineer.create_features(sample_data)
    
    # Check output
    assert len(features) > 0
    assert 'recency_days' in features.columns
    assert 'frequency' in features.columns
    assert 'monetary_total' in features.columns
    assert 'is_churned' in features.columns
    
    # Check no NaN values in key features
    assert features['recency_days'].notna().all()
    assert features['frequency'].notna().all()


def test_model_training(sample_data):
    """Test model training."""
    # Create features
    engineer = ChurnFeatureEngineer(churn_threshold_days=90)
    features = engineer.create_features(sample_data)
    
    # Prepare data
    X = features[engineer.get_feature_names()]
    y = features['is_churned']
    
    # Train model
    predictor = ChurnPredictor(use_smote=False, random_state=42)
    metrics = predictor.train(X, y, test_size=0.3)
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1


def test_model_prediction(sample_data):
    """Test model predictions."""
    # Create features
    engineer = ChurnFeatureEngineer(churn_threshold_days=90)
    features = engineer.create_features(sample_data)
    
    # Train model
    X = features[engineer.get_feature_names()]
    y = features['is_churned']
    
    predictor = ChurnPredictor(use_smote=False, random_state=42)
    predictor.train(X, y, test_size=0.3)
    
    # Make predictions
    predictions = predictor.predict(X)
    probabilities = predictor.predict_proba(X)
    
    # Check outputs
    assert len(predictions) == len(X)
    assert len(probabilities) == len(X)
    assert all(p in [0, 1] for p in predictions)
    assert all(0 <= p <= 1 for p in probabilities)


def test_customer_risk_assessment(sample_data):
    """Test customer risk assessment."""
    engineer = ChurnFeatureEngineer(churn_threshold_days=90)
    features = engineer.create_features(sample_data)
    
    X = features[engineer.get_feature_names()]
    y = features['is_churned']
    X['CustomerID'] = features['CustomerID']
    
    predictor = ChurnPredictor(use_smote=False, random_state=42)
    predictor.train(X.drop('CustomerID', axis=1), y, test_size=0.3)
    
    # Get risk assessment
    risk = predictor.get_customer_risk(X)
    
    # Check output
    assert 'CustomerID' in risk.columns
    assert 'churn_probability' in risk.columns
    assert 'churn_prediction' in risk.columns
    assert 'risk_level' in risk.columns
    
    # Check risk levels
    valid_risk_levels = {'Low', 'Medium', 'High'}
    assert all(level in valid_risk_levels for level in risk['risk_level'])
