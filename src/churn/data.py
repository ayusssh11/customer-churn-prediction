
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
)


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and clean the Telco customer churn dataset."""
    df = pd.read_csv(path)

    required_columns = {"customerID", "TotalCharges", "Churn"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    # Convert blank or whitespace-only TotalCharges values to NaN,
    # then convert the column to numeric.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(r"^\s*$", float("nan"), regex=True),
        errors="coerce",
    )

    # Remove rows where the target or required numeric value is missing.
    df = df.dropna(subset=["Churn", "TotalCharges"]).copy()

    # Remove the identifier; it should not be used as a model feature.
    df = df.drop(columns=["customerID"])

    return df


def split_features_target(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split the cleaned data into training and test sets."""
    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a 'Churn' target column.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Churn contains values other than 'Yes' and 'No'.")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
