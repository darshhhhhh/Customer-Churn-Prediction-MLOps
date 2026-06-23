import yaml
import joblib
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_artifacts():
    config = load_config()

    model_path = config["model"]["model_path"]

    model = joblib.load(model_path)
    label_encoders = joblib.load("models/label_encoders.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    return model, label_encoders, feature_columns


def preprocess_input(customer_data, label_encoders, feature_columns):
    input_df = pd.DataFrame([customer_data])

    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            try:
                input_df[column] = encoder.transform(input_df[column])
            except ValueError as error:
                raise ValueError(
                    f"Invalid value for column '{column}'. "
                    f"Allowed values are: {list(encoder.classes_)}"
                ) from error

    input_df = input_df[feature_columns]

    return input_df


def predict_churn(customer_data):
    model, label_encoders, feature_columns = load_artifacts()

    processed_input = preprocess_input(
        customer_data,
        label_encoders,
        feature_columns
    )

    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    result = {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4)
    }

    return result


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 425.5
    }

    result = predict_churn(sample_customer)

    print("\nCustomer Churn Prediction")
    print("-------------------------")
    print(f"Prediction: {result['prediction']}")
    print(f"Churn Probability: {result['churn_probability']}")