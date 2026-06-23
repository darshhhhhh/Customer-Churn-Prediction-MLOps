import os
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost as mlflow_xgboost
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_processed_data():
    train_data = pd.read_csv("data/processed/train.csv")
    test_data = pd.read_csv("data/processed/test.csv")
    return train_data, test_data


def split_features_target(train_data, test_data, target_column):
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    return metrics


def train_models():
    config = load_config()

    target_column = config["training"]["target_column"]
    model_path = config["model"]["model_path"]
    random_state = config["model"]["random_state"]

    train_data, test_data = load_processed_data()

    X_train, X_test, y_train, y_test = split_features_target(
        train_data,
        test_data,
        target_column
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss"
        )
    }

    mlflow.set_experiment("Customer Churn Prediction")

    best_model = None
    best_model_name = None
    best_f1_score = 0
    best_metrics = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_param("model_name", model_name)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if model_name == "XGBoost":
                mlflow_xgboost.log_model(
                    xgb_model=model,
                    name="model"
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model"
                )

            print(f"\nModel: {model_name}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")

            if metrics["f1_score"] > best_f1_score:
                best_f1_score = metrics["f1_score"]
                best_model = model
                best_model_name = model_name
                best_metrics = metrics

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(best_model, model_path)

    with open("reports/metrics.json", "w") as file:
        json.dump(
            {
                "best_model": best_model_name,
                "metrics": best_metrics
            },
            file,
            indent=4
        )

    print("\nTraining completed successfully.")
    print(f"Best Model: {best_model_name}")
    print(f"Best F1 Score: {best_f1_score:.4f}")
    print(f"Model saved at: {model_path}")
    print("Metrics saved at: reports/metrics.json")


if __name__ == "__main__":
    train_models()