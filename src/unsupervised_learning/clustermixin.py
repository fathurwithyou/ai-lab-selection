from abc import ABC, abstractmethod


class ClusterMixin(ABC):
    """
    A mixin class for clustering algorithms providing a common interface.

    Provides a default `fit_predict` that either returns `labels_` set during
    `fit` or, if available, defers to an implemented `predict` method.
    """

    @abstractmethod
    def fit(self, X):
        """
        Fit the clustering model to the data.

        Parameters
        ----------
        X : array-like
            Input data to cluster.
        """
        pass

    def fit_predict(self, X):
        """
        Fit the model on X and return cluster labels.

        If the estimator sets `labels_` during `fit`, those are returned.
        Otherwise, if a `predict` method is available, it is used.
        """
        self.fit(X)

        if hasattr(self, "labels_") and self.labels_ is not None:
            return self.labels_

        if hasattr(self, "predict") and callable(self.predict):
            return self.predict(X)

        raise AttributeError(
            "ClusterMixin.fit_predict requires the estimator to set `labels_` during "
            "fit or implement a `predict` method."
        )
