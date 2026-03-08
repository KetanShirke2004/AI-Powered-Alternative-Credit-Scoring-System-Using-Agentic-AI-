"""
utils/named_pipeline.py

Shared module so NamedPipeline can be pickled in train_model.py
and unpickled in data_utils.py without a "Can't get attribute" error.

IMPORTANT: Both files must import from THIS module:
    from utils.named_pipeline import NamedPipeline
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class NamedPipeline(BaseEstimator, TransformerMixin):
    """
    Thin wrapper around sklearn Pipeline that stores feature_names_in_
    as a plain instance attribute (sklearn Pipeline disallows the setter).

    Pickle-safe: as long as this module is on the Python path,
    joblib.load() can reconstruct the object in any script.
    """

    def __init__(self, steps):
        self.steps             = steps
        self._pipeline         = Pipeline(steps)
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self._pipeline.fit(X, y)
        self.feature_names_in_ = np.array(
            X.columns.tolist() if hasattr(X, "columns") else list(range(X.shape[1]))
        )
        return self

    def transform(self, X):
        return self._pipeline.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def __repr__(self):
        n = len(self.feature_names_in_) if self.feature_names_in_ is not None else "?"
        return f"NamedPipeline(steps={[s[0] for s in self.steps]}, features={n})"