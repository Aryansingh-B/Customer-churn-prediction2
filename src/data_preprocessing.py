import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def load_raw_data(filepath: str = "data/telco_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges has spaces instead of NaN — fix it
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill NaN TotalCharges with 0 (new customers with 0 tenure)
    df["TotalCharges"].fillna(0, inplace=True)

    # Encode target: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avg monthly spend
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Customer lifetime value proxy
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]

    # Number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]),
        axis=1
    )

    # Tenure buckets (useful for segmentation in dashboard)
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
        include_lowest=True
    )

    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()

    # Separate tenure bucket before encoding (keep for dashboard display)
    tenure_bucket = df["TenureBucket"].copy()
    df.drop(columns=["TenureBucket"], inplace=True)

    # Binary columns — map directly
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # Multi-category columns — one-hot encode
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Convert all bool columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df, tenure_bucket


def get_train_test_split(df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, save_path: str = "models/scaler.pkl"):
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "CLV", "NumServices"]
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to {save_path}")

    return X_train, X_test, scaler


def run_preprocessing_pipeline(filepath: str = "data/telco_churn.csv"):
    print("Loading data...")
    df = load_raw_data(filepath)
    print(f"Shape: {df.shape}")

    print("Cleaning...")
    df = clean_data(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Encoding...")
    df, tenure_bucket = encode_features(df)

    print("Splitting...")
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print("Scaling...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
    print(f"Churn rate (train): {y_train.mean():.2%}")

    return X_train, X_test, y_train, y_test, scaler, df


if __name__ == "__main__":
    run_preprocessing_pipeline()