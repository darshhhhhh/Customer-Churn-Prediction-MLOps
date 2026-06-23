import os
import json
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(model_path):
    return joblib.load(model_path)


def load_test_data(target_column):
    test_data = pd.read_csv("data/processed/test.csv")

    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    return X_test, y_test


def evaluate_saved_model():
    config = load_config()

    target_column = config["training"]["target_column"]
    model_path = config["model"]["model_path"]

    model = load_model(model_path)
    X_test, y_test = load_test_data(target_column)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("reports", exist_ok=True)

    with open("reports/evaluation_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    with open("reports/classification_report.json", "w") as file:
        json.dump(report, file, indent=4)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    display.plot()
    plt.title("Confusion Matrix - Customer Churn Prediction")
    plt.savefig("reports/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    print("\nModel Evaluation Completed Successfully.")

    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nSaved files:")
    print("reports/evaluation_metrics.json")
    print("reports/classification_report.json")
    print("reports/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_saved_model()