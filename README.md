# Customer Churn Prediction with MLOps

An end-to-end Machine Learning and MLOps project that predicts whether a customer is likely to churn based on demographic, account, service, and billing information.

The project includes data preprocessing, model training, experiment tracking with MLflow, model evaluation, FastAPI deployment, Streamlit UI, Docker containerization, and GitHub Actions CI.

---

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using a service. This project uses machine learning to classify customers as either:

* Churn
* No Churn

The goal is to build a production-style ML pipeline instead of only a notebook-based model.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* MLflow
* FastAPI
* Streamlit
* Docker
* GitHub Actions
* Pytest

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_app.py
│   └── test_project_structure.py
│
├── .dockerignore
├── .gitignore
├── config.yaml
├── Dockerfile
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Dataset

This project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

* Gender
* Senior citizen status
* Partner and dependent status
* Tenure
* Phone service
* Internet service
* Online security
* Online backup
* Device protection
* Tech support
* Streaming services
* Contract type
* Billing method
* Monthly charges
* Total charges
* Churn status

Place the dataset in:

```text
data/raw/Telco-Customer-Churn.csv
```

---

## Model Performance

The best model selected during training was Logistic Regression based on F1 score.

Evaluation metrics:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.7991 |
| Precision | 0.6426 |
| Recall    | 0.5481 |
| F1 Score  | 0.5916 |
| ROC-AUC   | 0.8403 |

Confusion matrix:

```text
[[921 114]
 [169 205]]
```

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/darshhhhhh/Customer-Churn-Prediction-MLOps
cd Customer-Churn-Prediction-MLOps
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the dataset file inside:

```text
data/raw/Telco-Customer-Churn.csv
```

### 5. Run data preprocessing

```bash
python src/data_preprocessing.py
```

This creates train and test datasets inside:

```text
data/processed/
```

It also saves label encoders and feature columns inside:

```text
models/
```

### 6. Train the model

```bash
python src/train.py
```

This trains multiple models and saves the best model at:

```text
models/churn_model.pkl
```

It also logs experiments using MLflow.

### 7. Evaluate the model

```bash
python src/evaluate.py
```

This generates evaluation files inside:

```text
reports/
```

---

## Single Customer Prediction

Run:

```bash
python src/predict.py
```

Example output:

```text
Customer Churn Prediction
-------------------------
Prediction: Churn
Churn Probability: 0.6739
```

---

## Run FastAPI App

Start the API server:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Example request body:

```json
{
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
```

Example response:

```json
{
  "prediction": "Churn",
  "churn_probability": 0.6739
}
```

---

## Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

This opens a simple web interface where users can enter customer details and get a churn prediction.

---

## Run with Docker

Build the Docker image:

```bash
docker build -t customer-churn-api .
```

Run the Docker container:

```bash
docker run -p 8000:8000 customer-churn-api
```

Open:

```text
http://127.0.0.1:8000
```

---

## Run Tests

```bash
pytest
```

Expected result:

```text
2 passed
```

---

## GitHub Actions CI

This project includes a GitHub Actions workflow that runs automatically on push or pull request.

The workflow checks:

* Python setup
* Dependency installation
* Python syntax
* FastAPI home endpoint test
* Required project files

Workflow file:

```text
.github/workflows/ci.yml
```

---

## Important Note

The following files are not pushed to GitHub because they are generated locally:

```text
data/raw/
data/processed/
models/*.pkl
models/*.joblib
reports/*.png
reports/*.json
mlruns/
mlartifacts/
venv/
```

To reproduce the project, run:

```bash
python src/data_preprocessing.py
python src/train.py
python src/evaluate.py
```

---

## Future Improvements

* Add model registry support with MLflow
* Add cloud deployment
* Add monitoring for model drift
* Add automated retraining pipeline
* Add advanced feature engineering
* Improve churn recall using class balancing techniques

---

## Author

Darshil Vaja
Machine Learning / MLOps Project
