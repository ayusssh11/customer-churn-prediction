
from churn.train import train_model


def test_train_model_returns_fitted_pipeline():
    model = train_model()

    assert hasattr(model, "predict")
    assert hasattr(model, "named_steps")
