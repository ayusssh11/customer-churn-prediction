
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.data import load_data, split_features_target
from churn.predict import predict_churn
from churn.train import build_pipeline


def test_load_data_has_expected_target():
    df = load_data()

    assert not df.empty
    assert "Churn" in df.columns
    assert "customerID" not in df.columns
    assert set(df["Churn"].unique()) == {"Yes", "No"}
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_split_features_target():
    df = load_data()
    X_train, X_test, y_train, y_test = split_features_target(df)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(y_train.unique()).issubset({0, 1})


def test_build_pipeline_returns_pipeline():
    df = load_data()
    X_train, _, _, _ = split_features_target(df)

    pipeline = build_pipeline(X_train)

    assert isinstance(pipeline, Pipeline)


def test_prediction_returns_valid_result():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
    }

    result = predict_churn(customer)

    assert result["prediction"] in {"Churn", "Stay"}
    assert 0.0 <= result["churn_probability"] <= 1.0
