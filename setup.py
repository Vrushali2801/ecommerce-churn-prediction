from setuptools import setup, find_packages

setup(
    name="online_retail_analysis",
    version="0.1.0",
    description="Online Retail Analysis with Churn Prediction",
    author="Vrushali",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.3",
        "sqlalchemy>=2.0.23",
        "psycopg2-binary>=2.9.9",
        "fastapi>=0.108.0",
        "uvicorn>=0.25.0",
        "mlflow>=2.9.2",
    ],
)
