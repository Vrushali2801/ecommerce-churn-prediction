# Customer Analytics & Churn Prediction Platform

AI-powered customer retention system with churn prediction, RFM segmentation, and Customer Lifetime Value (CLV) analysis.

## 🎯 Overview

An end-to-end analytics platform built for e-commerce businesses to identify at-risk customers and optimize retention strategies. The system processes 391K+ transactions from 4,319 customers, achieving 95%+ prediction accuracy.

**Key Capabilities:**
- 🤖 ML-powered churn prediction (Random Forest with SMOTE)
- 📊 RFM customer segmentation (9 segments from Champions to At Risk)
- 💰 Customer Lifetime Value (CLV) projections
- 🚀 Production-ready REST API with FastAPI
- 📈 MLflow experiment tracking
- 🗄️ PostgreSQL data warehouse
- 🎨 Interactive web UI for customer lookup

## ⚡ Quick Start

### Setup (5 minutes)

1. **Activate environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Configure database:**
   Create `.env` file with your PostgreSQL credentials:
   ```env
   DB_HOST=localhost
   DB_PORT=5433
   DB_NAME=online_retail
   DB_USER=postgres
   DB_PASSWORD=your_password_here
   ```

3. **Train model:**
   ```powershell
   # Terminal 1: Start MLflow
   mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 127.0.0.1 --port 5000
   
   # Terminal 2: Train
   python mlflow/train_churn.py
   ```
   Expected: Model saved to `models/churn_model.pkl` with ROC-AUC ~0.88

4. **Start API:**
   ```powershell
   uvicorn api.main:app --reload --port 8000
   ```

5. **Open UI:**
   Navigate to `ui/index.html` in your browser and enter a customer ID (e.g., 17850)

## 🔧 Tech Stack

- **Backend:** Python 3.10, FastAPI, scikit-learn
- **Database:** PostgreSQL 15
- **ML Ops:** MLflow, joblib
- **Frontend:** HTML/CSS/JavaScript (Vanilla)
- **Data:** pandas, numpy


**Access MLflow UI:** http://localhost:5000

### 2. Start the API

```powershell
# From the project root
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access:**
- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
## 📡 API Endpoints

**Main Endpoints:**
- `GET /api/v1/customer/lookup/{customer_id}` - Get complete customer analysis (churn + RFM + CLV)
- `POST /api/v1/churn/predict` - Predict churn for custom features
- `GET /api/v1/rfm/segments` - View all RFM segments
- `GET /api/v1/health` - API health check

**Interactive Docs:** http://localhost:8000/docs

## 💡 Usage Examples

### Customer Lookup (Recommended)
```powershell
# Using the UI
Open ui/index.html → Enter Customer ID: 17850

# Using API directly
curl http://localhost:8000/api/v1/customer/lookup/17850
```

**Response includes:**
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

### Batch Churn Prediction
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
        "monetary_std": 30,
        "total_items": 40,
        "avg_items_per_order": 5,
        "max_items_per_order": 10,
        "avg_unit_price": 30,
        "max_unit_price": 75,
        "min_unit_price": 10,
        "price_range": 65,
        "tenure_days": 200,
        "active_days": 180,
        "purchase_velocity": 1.2,
        "avg_days_between_purchases": 25
    }
}

response = requests.post(url, json=data)
print(response.json())
```

## 📊 Model Performance

- **Algorithm:** Random Forest Classifier with SMOTE
- **Features:** 16 engineered features (RFM + behavioral + temporal)
- **Accuracy:** 95%+ on test set
- **ROC-AUC:** 0.88
- **Training Data:** 391,925 transactions from 4,319 customers
- **Tracking:** MLflow for experiment management

## 🗂️ Data Pipeline

```
Raw Data (PostgreSQL) 
  ↓
Data Loader → Feature Engineering (16 features)
  ↓
ML Models (Churn + RFM + CLV)
  ↓
FastAPI Endpoints
  ↓
JSON Response / Web UI
```

## 🛠️ Troubleshooting

**Port already in use:**
```powershell
# Find and kill process on port 8000
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

**Database connection error:**
- Verify PostgreSQL is running on port 5433
- Check `.env` credentials match your database

**Model not found:**
```powershell
# Retrain model
python mlflow/train_churn.py
```

## 📈 Key Features

**RFM Segmentation (9 Segments):**
- Champions, Loyal Customers, Potential Loyalists
- New Customers, Promising, Need Attention
- About to Sleep, At Risk, Others

**Churn Risk Levels:**
- Low Risk: 0-30% churn probability
- Medium Risk: 30-70% churn probability  
- High Risk: 70-100% churn probability

**CLV Projections:**
- 1-year customer lifetime value forecast
- Segments: Low, Medium, High, VIP

## 📄 License

Educational and analytical purposes.
