cat > README.md <<'EOF'
# Customer Churn Prediction

A Python machine learning project demonstrating a reproducible
training workflow using scikit-learn.

> Current status: The initial pipeline uses scikit-learn's built-in
> breast cancer dataset to validate model training and testing.
> Customer churn data and churn-specific evaluation have not yet
> been implemented.

## Features

- Python project managed with uv
- Reproducible dependency resolution with uv.lock
- Scikit-learn preprocessing and classification pipeline
- Train/test split with stratification
- Model evaluation using accuracy and a classification report
- Model serialization with joblib
- Automated tests with pytest
- Code quality checks with Ruff

## Requirements

- Python 3.13 or a compatible version
- uv

## Setup

Clone the repository:

    git clone <YOUR_REPOSITORY_URL>
    cd customer-churn-prediction

Install dependencies:

    uv sync

## Train the model

    uv run python -m churn.train

The script trains a logistic regression pipeline and saves the
model artifact under models/.

## Run tests

    uv run pytest -v

## Code quality

Check code:

    uv run ruff check .

Format code:

    uv run ruff format .

## Project structure

    src/churn/     Reusable ML code
    tests/         Automated tests
    notebooks/     Experiments and exploration
    data/          Local datasets
    models/        Generated model artifacts

## Reproducibility

The project uses a fixed random seed for the train/test split.
Dependencies are recorded in pyproject.toml and uv.lock.

## Limitations

This is an educational ML pipeline, not a production deployment.
The current example uses a breast cancer classification dataset,
not customer churn data. Reported metrics should not be interpreted
as business churn performance.

## License

No license has been selected yet. All rights reserved unless a
license is added to this repository.
EOF