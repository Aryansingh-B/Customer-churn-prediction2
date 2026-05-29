import pandas as pd 
import numpy as np 
import json 
import joblib
import os 
from dotenv import load_dotenv

load_dotenv()



def load_model(model_path: str = "models/xgb_churn_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.model_training"
        )
    return joblib.load(model_path)


def load_scaler(scaler_path: str = "models/scaler.pkl"):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    return joblib.load(scaler_path)


def load_feature_names(path: str = "models/feature_names.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_metrics(path: str = "models/metrics.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_feature_importance(path: str = "models/feature_importance.csv"):
    return pd.read_csv(path)


def load_test_predictions(path: str = "models/test_predictions.csv"):
    return pd.read_csv(path)


def get_churn_risk_label(prob: float) -> tuple:
    """Returns (label, color) for a churn probability."""
    if prob >= 0.70:
        return "🔴 High Risk", "#FF4B4B"
    elif prob >= 0.40:
        return "🟡 Medium Risk", "#FFA500"
    else:
        return "🟢 Low Risk", "#00C851"


def calculate_clv_impact(df: pd.DataFrame) -> dict:
    """Business metric: revenue at risk from churners."""
    churners = df[df["Predicted_Churn"] == 1]
    non_churners = df[df["Predicted_Churn"] == 0]

    # Rough CLV = monthly charges × avg remaining tenure (assume 12 months)
    clv_at_risk = (churners["MonthlyCharges"].sum() * 12
                   if "MonthlyCharges" in churners.columns else 0)

    return {
        "total_customers": len(df),
        "predicted_churners": len(churners),
        "churn_rate": len(churners) / len(df),
        "clv_at_risk": clv_at_risk,
        "avg_churn_prob": df["Churn_Probability"].mean()
    }


def get_app_title() -> str:
    return os.getenv("APP_TITLE", "Customer Churn Predictor")
