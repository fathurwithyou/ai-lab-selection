import numpy as np

from ..base_estimator import BaseEstimator
from .classifier import ClassifierMixin


class GaussianNaiveBayes(ClassifierMixin, BaseEstimator):
    """
    Gaussian Naive Bayes classifier.

    This classifier assumes that the features follow a Gaussian (normal) distribution
    within each class and that features are conditionally independent given the class.
    """

    def __init__(self, var_smoothing=1e-9):
        """
        Initialize Gaussian Naive Bayes classifier.

        Parameters:
        var_smoothing (float): Portion of the largest variance of all features
            that is added to variances for calculation stability.
        """
        self.var_smoothing = var_smoothing
        self.classes = None
        self.class_count = None
        self.class_prior = None
        self.theta = None
        self.sigma = None
        self.epsilon = None

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes classifier.

        Parameters:
        X (array-like): Training features of shape (n_samples, n_features)
        y (array-like): Target labels of shape (n_samples,)

        Returns:
        self: Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)

        self.classes, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_count = counts
        self.class_prior = counts / len(y)

        self.theta = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features))

        for i, class_label in enumerate(self.classes):
            class_mask = y == class_label
            X_class = X[class_mask]

            self.theta[i] = np.mean(X_class, axis=0)
            self.sigma[i] = np.var(X_class, axis=0)

        self.epsilon = self.var_smoothing * np.var(X, axis=0).max()
        self.sigma += self.epsilon

        return self

    def _calculate_log_likelihood(self, X):
        """
        Calculate log likelihood for each class.

        Parameters:
        X (array-like): Features to calculate likelihood for

        Returns:
        log_likelihood (array): Log likelihood for each class
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        log_likelihood = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            log_norm = -0.5 * np.sum(np.log(2 * np.pi * self.sigma[i]))

            diff_squared = (X - self.theta[i]) ** 2
            log_exp = -0.5 * np.sum(diff_squared / self.sigma[i], axis=1)

            log_likelihood[:, i] = log_norm + log_exp

        return log_likelihood

    def _calculate_class_log_prior(self):
        """Calculate log of class priors."""
        return np.log(self.class_prior)

    def predict_log_proba(self, X):
        """
        Return log-probabilities for each class.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        log_proba (array): Log probabilities of shape (n_samples, n_classes)
        """
        X = np.array(X)

        if self.classes is None:
            raise ValueError("Model must be fitted before making predictions")

        log_likelihood = self._calculate_log_likelihood(X)

        log_prior = self._calculate_class_log_prior()
        log_posterior = log_likelihood + log_prior

        log_marginal = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        log_proba = log_posterior - log_marginal

        return log_proba

    def predict_proba(self, X):
        """
        Return probabilities for each class.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        proba (array): Probabilities of shape (n_samples, n_classes)
        """
        log_proba = self.predict_log_proba(X)
        return np.exp(log_proba)

    def predict(self, X):
        """
        Predict class labels.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        predictions (array): Predicted class labels of shape (n_samples,)
        """
        log_proba = self.predict_log_proba(X)
        predicted_indices = np.argmax(log_proba, axis=1)
        return self.classes[predicted_indices]

    def score(self, X, y):
        """
        Calculate accuracy score.

        Parameters:
        X (array-like): Test features
        y (array-like): True labels

        Returns:
        accuracy (float): Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
