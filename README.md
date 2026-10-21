# Customer Churn Prediction

A reproducible machine learning project built with Python, scikit-learn, and `uv`.

> **Current status:** The initial training pipeline uses scikit-learn's built-in breast cancer dataset to validate the model training and testing workflow. Customer churn data and churn-specific evaluation are future improvements.

## Features

- Reproducible Python environment using `uv`
- Dependency locking with `uv.lock`
- Scikit-learn preprocessing and classification pipeline
- Stratified train/test split
- Model evaluation with accuracy and classification metrics
- Model serialization with joblib
- Automated tests with pytest
- Code quality checks with Ruff

## Requirements

- Python 3.13 or a compatible version
- [uv](https://docs.astral.sh/uv/)

## Getting Started

Clone the repository:

```bash
git clone https://github.com/ayusssh11/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:

```bash
uv sync
```

## Train the Model

```bash
uv run python -m churn.train
```

The training script evaluates the model and saves the trained artifact under `models/`.

## Run Tests

```bash
uv run pytest -v
```

## Check Code Quality

```bash
uv run ruff check .
uv run ruff format --check .
```

## Project Structure

```text
src/churn/   Reusable ML code
tests/       Automated tests
notebooks/   Experiments and analysis
data/        Local datasets
models/      Generated model artifacts
```

## Reproducibility

Dependencies are declared in `pyproject.toml` and resolved versions are recorded in `uv.lock`. The training workflow uses a fixed random seed for its train/test split.

## Limitations

This is an educational project, not a production deployment or medical diagnostic tool. The current dataset is for breast cancer classification and does not represent customer churn. Churn-specific data, metrics, and validation remain to be implemented.

## License

No open-source license has been selected yet. All rights reserved unless a license is added.
