import pandas as pd
import numpy as np
import joblib
import os
import json
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE

from src.data_preprocessing import run_preprocessing_pipeline


def apply_smote(X_train, y_train):
    """Handle class imbalance — churn is ~26% in this dataset."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE → Churn rate: {y_res.mean():.2%}")
    return X_res, y_res


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print(f"\n{'='*40}")
    print(f"{model_name} Results")
    print(f"{'='*40}")
    print(f"ROC-AUC  : {metrics['roc_auc']}")
    print(f"F1 Score : {metrics['f1_score']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall   : {metrics['recall']}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return metrics, y_prob


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return fi_df


def save_artifacts(model, metrics, feature_names, X_train_columns):
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/xgb_churn_model.pkl")
    print("Model saved → models/xgb_churn_model.pkl")

    # Save feature names (needed for Streamlit app)
    with open("models/feature_names.json", "w") as f:
        json.dump(list(X_train_columns), f)
    print("Feature names saved → models/feature_names.json")

    # Save metrics
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved → models/metrics.json")

    # Save feature importance
    fi_df = get_feature_importance(model, feature_names)
    fi_df.to_csv("models/feature_importance.csv", index=False)
    print("Feature importance saved → models/feature_importance.csv")


def main():
    # 1. Preprocess
    X_train, X_test, y_train, y_test, scaler, full_df = run_preprocessing_pipeline()

    # 2. Handle imbalance
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 3. Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train_res, y_train_res)
    xgb_metrics, xgb_probs = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # 4. Train Logistic Regression for comparison
    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train_res, y_train_res)
    lr_metrics, lr_probs = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # 5. Save best model (XGBoost wins almost always)
    save_artifacts(xgb_model, xgb_metrics, X_train.columns.tolist(), X_train.columns)

    # 6. Save test predictions for dashboard
    test_results = X_test.copy()
    test_results["Actual_Churn"] = y_test.values
    test_results["Churn_Probability"] = xgb_probs
    test_results["Predicted_Churn"] = (xgb_probs >= 0.5).astype(int)
    test_results.to_csv("models/test_predictions.csv", index=False)
    print("Test predictions saved → models/test_predictions.csv")

    print("\nAll artifacts saved. Ready to launch dashboard!")


if __name__ == "__main__":
    main()