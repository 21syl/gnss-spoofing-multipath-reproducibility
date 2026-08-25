"""DEMO PLACEHOLDER for leakage-safe feature preprocessing."""


def fit_normalizer_on_training_records(train_records):
    """Fit preprocessing parameters only on training records."""
    raise NotImplementedError("DEMO_ONLY")


def transform_records(records, fitted_normalizer):
    """Apply the frozen training-set transformation to validation or test records."""
    raise NotImplementedError("DEMO_ONLY")
