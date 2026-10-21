
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"


def predict_churn(customer: dict) -> dict:
    """Predict churn for one customer using the saved pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `uv run python -m churn.train` first."
        )

    model = joblib.load(MODEL_PATH)
    customer_df = pd.DataFrame([customer])

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = int(probability >= 0.5)

    return {
        "prediction": "Churn" if prediction else "Stay",
        "churn_probability": round(probability, 4),
    }


if __name__ == "__main__":
    sample_customer = {
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

    result = predict_churn(sample_customer)
    print("Prediction:", result["prediction"])
    print("Churn probability:", f"{result['churn_probability']:.1%}")
