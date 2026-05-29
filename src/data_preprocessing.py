import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def load_raw_data(filepath: str = "data/telco_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Drop columns that are IDs, geo, or leakage ──────────────────────────
    # Churn Score & Churn Reason are post-churn leakage — must drop
    # CLTV we'll keep as a feature
    drop_cols = [
        "CustomerID", "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Churn Score", "Churn Reason", "Churn Label"  # keep Churn Value as target
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    print(f"Dropped: {drop_cols}")

    # ── Rename columns: remove spaces, standardize ───────────────────────────
    df.rename(columns={
        "Churn Value"       : "Churn",
        "Tenure Months"     : "tenure",
        "Monthly Charges"   : "MonthlyCharges",
        "Total Charges"     : "TotalCharges",
        "Senior Citizen"    : "SeniorCitizen",
        "Phone Service"     : "PhoneService",
        "Multiple Lines"    : "MultipleLines",
        "Internet Service"  : "InternetService",
        "Online Security"   : "OnlineSecurity",
        "Online Backup"     : "OnlineBackup",
        "Device Protection" : "DeviceProtection",
        "Tech Support"      : "TechSupport",
        "Streaming TV"      : "StreamingTV",
        "Streaming Movies"  : "StreamingMovies",
        "Paperless Billing" : "PaperlessBilling",
        "Payment Method"    : "PaymentMethod",
        "CLTV"              : "CLTV_given",   # already in dataset
    }, inplace=True)

    # ── Fix TotalCharges (spaces → NaN) ─────────────────────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)

    # ── Target is already 0/1 (Churn Value) ─────────────────────────────────
    df["Churn"] = df["Churn"].astype(int)

    print(f"After clean shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avg monthly spend
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # CLV proxy (if not already present from dataset)
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]

    # Number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(
            1 for v in row
            if v not in ["No", "No internet service", "No phone service"]
        ),
        axis=1
    )

    # Tenure buckets (for dashboard segmentation display)
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
        include_lowest=True
    )

    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()

    # Separate TenureBucket before encoding
    tenure_bucket = df["TenureBucket"].copy()
    df.drop(columns=["TenureBucket"], inplace=True)

    # ── Binary columns ───────────────────────────────────────────────────────
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_cols = ["Gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    binary_cols = [c for c in binary_cols if c in df.columns]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # SeniorCitizen — already Yes/No string in this dataset version
    if df["SeniorCitizen"].dtype == object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map(binary_map)

    # ── One-hot encode multi-category columns ────────────────────────────────
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Convert booleans to int
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
    numeric_cols = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "AvgMonthlySpend", "CLV", "NumServices", "CLTV_given"
    ]
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved → {save_path}")

    return X_train, X_test, scaler


def run_preprocessing_pipeline(filepath: str = "data/telco_churn.csv"):
    print("\n── Loading ──────────────────────────────")
    df = load_raw_data(filepath)

    print("\n── Cleaning ─────────────────────────────")
    df = clean_data(df)

    print("\n── Engineering Features ─────────────────")
    df = engineer_features(df)

    print("\n── Encoding ─────────────────────────────")
    df, tenure_bucket = encode_features(df)

    print("\n── Splitting ────────────────────────────")
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print("\n── Scaling ──────────────────────────────")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
    print(f"Churn rate (train): {y_train.mean():.2%}")

    return X_train, X_test, y_train, y_test, scaler, df


if __name__ == "__main__":
    run_preprocessing_pipeline()