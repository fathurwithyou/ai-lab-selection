from abc import ABC, abstractmethod


class ClassifierMixin(ABC):
    """
    A mixin class for classifiers that provides a common interface for training and prediction."""

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the classifier to the training data.

        Parameters:
        X (array-like): Training data features.
        y (array-like): Target labels.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the labels for the given data.

        Parameters:
        X (array-like): Data features to predict labels for.
        Returns:
        array-like: Predicted labels.
        """
        pass