import os
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_data(raw_path):
    return pd.read_csv(raw_path)


def clean_data(df):
    df = df.copy()

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def encode_data(df):
    df = df.copy()
    label_encoders = {}

    for column in df.columns:
        if df[column].dtype == "object":
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            label_encoders[column] = encoder

    return df, label_encoders


def preprocess_data():
    config = load_config()

    raw_path = config["data"]["raw_path"]
    target_column = config["training"]["target_column"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]

    df = load_data(raw_path)
    df = clean_data(df)

    df, label_encoders = encode_data(df)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_data = X_train.copy()
    train_data[target_column] = y_train

    test_data = X_test.copy()
    test_data[target_column] = y_test

    train_data.to_csv("data/processed/train.csv", index=False)
    test_data.to_csv("data/processed/test.csv", index=False)

    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    print("Data preprocessing completed successfully.")
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    print("Label encoders saved at: models/label_encoders.pkl")
    print("Feature columns saved at: models/feature_columns.pkl")


if __name__ == "__main__":
    preprocess_data()