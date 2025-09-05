import numpy as np

import sys
sys.path.append('..')
from base_estimator import BaseEstimator


class PCA(BaseEstimator):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_in_ = None
        self.n_samples_ = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.n_samples_, self.n_features_in_ = X.shape

        if self.n_components is None:
            self.n_components_ = min(self.n_samples_, self.n_features_in_)
        else:
            self.n_components_ = min(
                self.n_components, self.n_samples_, self.n_features_in_
            )

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = s[: self.n_components_]

        self.explained_variance_ = (s[: self.n_components_] ** 2) / (
            self.n_samples_ - 1
        )
        total_var = np.sum((s**2) / (self.n_samples_ - 1))
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X):
        if self.components_ is None:
            raise ValueError("Model must be fitted before transforming data")

        X = np.array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PCA is expecting {self.n_features_in_} features"
            )

        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if self.components_ is None:
            raise ValueError("Model must be fitted before inverse transforming data")

        X = np.array(X, dtype=float)
        return np.dot(X, self.components_) + self.mean_
