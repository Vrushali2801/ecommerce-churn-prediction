# E-Commerce Churn Prediction

Predicting which customers are likely to churn using machine learning on real retail transaction data.

## What This Does

This project analyzes customer purchase behavior to predict churn risk. I worked with a dataset of ~391K transactions from 4,319 customers and built a Random Forest model that achieves 95%+ accuracy in identifying customers who might stop buying.

The system also includes RFM (Recency, Frequency, Monetary) segmentation to group customers into categories like "Champions" or "At Risk", plus Customer Lifetime Value predictions.

**What's included:**
- Churn prediction model (Random Forest with SMOTE for handling imbalanced data)
- RFM customer segmentation 
- CLV (Customer Lifetime Value) calculations
- REST API built with FastAPI
- Simple web interface for looking up customers
- MLflow for tracking experiments
- PostgreSQL database integration

## Getting Started

## Getting Started

### Setup

Activate your virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

Create a `.env` file with your database info:
```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=online_retail
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### Training the Model

First, start MLflow (keeps track of your experiments):
```powershell
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 127.0.0.1 --port 5000
```

In another terminal, train the model:
```powershell
python mlflow/train_churn.py
```

This will load the data, engineer features, train a Random Forest classifier, and save the model to `models/churn_model.pkl`. Should take a few minutes.

### Running the API

```powershell
uvicorn api.main:app --reload --port 8000
```

Then open `ui/index.html` in your browser and try looking up a customer (try ID 17850 - it's a high-risk customer).

API docs are at http://localhost:8000/docs

## Tech Stack

- Python 3.10
- FastAPI for the API
- scikit-learn for ML
- PostgreSQL for data storage
- MLflow for tracking experiments
- Vanilla JS for the UI (kept it simple)

## How to Use It

### Look up a customer

Open `ui/index.html` in your browser and enter a customer ID (try 17850).

You'll get back their churn risk, RFM segment, predicted lifetime value, and purchase history. For example, customer 17850 has a 90% churn probability and is classified as "At Risk" despite spending $5,391 over 34 purchases.

### Using the API directly

```bash
curl http://localhost:8000/api/v1/customer/lookup/17850
```

Response:
```json
{
  "customer_id": 17850,
  "churn_probability": 0.905,
  "risk_level": "High",
  "rfm_segment": "At Risk",
  "clv_1year": 1967791.45,
  "total_purchases": 34,
  "total_spent": 5391.30,
  "recency_days": 371
}
```

### Making predictions with custom data

If you want to test the model with your own feature values:

```python
import requests

url = "http://localhost:8000/api/v1/churn/predict"
data = {
    "customer_features": {
        "CustomerID": 12345,
        "recency_days": 45,
        "frequency": 8,
        "monetary_total": 1200,
        "monetary_mean": 150,
        # ... other features
    }
}

response = requests.post(url, json=data)
print(response.json())
```

Check the API docs at http://localhost:8000/docs for all available endpoints.

## About the Model

Using Random Forest with SMOTE to handle class imbalance (most customers don't churn, so the dataset was pretty skewed).

The model uses 16 engineered features:
- RFM metrics (recency, frequency, monetary)
- Purchase patterns (items per order, price ranges)
- Temporal features (tenure, active days, purchase velocity)

Results on test data:
- 95%+ accuracy
- ROC-AUC: 0.88
- Trained on 391,925 transactions from 4,319 customers

All experiments tracked in MLflow.

## How It Works

1. Data gets loaded from PostgreSQL
2. Feature engineering creates 16 features from raw transactions
3. Three models run in parallel:
   - Churn prediction (Random Forest)
   - RFM segmentation (rule-based scoring)
   - CLV calculation (revenue projections)
4. FastAPI serves everything via REST endpoints
5. Simple web UI displays results

## Common Issues

**Port 8000 already in use?**
```powershell
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

**Can't connect to database?**
Make sure PostgreSQL is running on port 5433 and your `.env` file has the right credentials.

**Model file not found?**
Train it first:
```powershell
python mlflow/train_churn.py
```

## What the Segments Mean

**RFM Segments:**
- Champions - Best customers, buy frequently
- Loyal Customers - Regular buyers
- At Risk - Haven't purchased in a while
- About to Sleep - Losing interest
- And 5 others...

**Risk Levels:**
- Low: 0-30% chance of churning
- Medium: 30-70%
- High: 70-100%

**CLV Tiers:**
- Projections for 1-year customer lifetime value
- Grouped into Low, Medium, High, VIP

## License

Built for learning and portfolio purposes.
