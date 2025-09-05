import numpy as np

from ..base_estimator import BaseEstimator
from .classifier import ClassifierMixin


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """
    Logistic Regression for binary classification using gradient descent or Newton's method.
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        regularization=None,
        reg_strength=0.01,
        optimizer="gradient_descent",
        tolerance=1e-6,
    ):
        """
        Initialize Logistic Regression classifier.

        Parameters:
        learning_rate (float): Learning rate for gradient descent
        max_iter (int): Maximum number of iterations
        regularization (str): 'l1', 'l2', or None
        reg_strength (float): Regularization strength
        optimizer (str): 'gradient_descent' or 'newton'
        tolerance (float): Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.optimizer = optimizer
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, X, y):
        """Compute logistic regression cost with regularization."""
        X.shape[0]
        z = X.dot(self.weights) + self.bias
        predictions = self._sigmoid(z)


        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)


        cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))


        if self.regularization == "l1":
            cost += self.reg_strength * np.sum(np.abs(self.weights))
        elif self.regularization == "l2":
            cost += self.reg_strength * np.sum(self.weights**2)

        return cost

    def _compute_gradients(self, X, y):
        """Compute gradients for weights and bias."""
        m = X.shape[0]
        z = X.dot(self.weights) + self.bias
        predictions = self._sigmoid(z)

        dw = (1 / m) * X.T.dot(predictions - y)
        db = (1 / m) * np.sum(predictions - y)


        if self.regularization == "l1":
            dw += self.reg_strength * np.sign(self.weights)
        elif self.regularization == "l2":
            dw += self.reg_strength * 2 * self.weights

        return dw, db

    def _compute_hessian(self, X):
        """Compute Hessian matrix for Newton's method."""
        m = X.shape[0]
        z = X.dot(self.weights) + self.bias
        predictions = self._sigmoid(z)


        S = predictions * (1 - predictions)


        H = (1 / m) * X.T.dot(np.diag(S)).dot(X)


        if self.regularization == "l2":
            H += self.reg_strength * 2 * np.eye(H.shape[0])

        return H

    def fit(self, X, y):
        """
        Fit the logistic regression model.

        Parameters:
        X (array-like): Training features of shape (n_samples, n_features)
        y (array-like): Binary target labels of shape (n_samples,)

        Returns:
        self: Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape


        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0
        self.cost_history = []

        for i in range(self.max_iter):

            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)

            if self.optimizer == "gradient_descent":

                dw, db = self._compute_gradients(X, y)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            elif self.optimizer == "newton":

                dw, db = self._compute_gradients(X, y)
                H = self._compute_hessian(X)

                try:

                    self.weights -= np.linalg.inv(H).dot(dw)
                    self.bias -= (
                        self.learning_rate * db
                    )
                except np.linalg.LinAlgError:

                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db


            if (
                i > 0
                and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance
            ):
                break

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        z = X.dot(self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        """Predict binary labels."""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


class SoftmaxRegression(ClassifierMixin, BaseEstimator):
    """
    Softmax Regression for multiclass classification.
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        regularization=None,
        reg_strength=0.01,
        optimizer="gradient_descent",
        tolerance=1e-6,
    ):
        """
        Initialize Softmax Regression classifier.

        Parameters:
        learning_rate (float): Learning rate for gradient descent
        max_iter (int): Maximum number of iterations
        regularization (str): 'l1', 'l2', or None
        reg_strength (float): Regularization strength
        optimizer (str): 'gradient_descent' or 'newton'
        tolerance (float): Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.optimizer = optimizer
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.classes = None
        self.cost_history = []

    def _softmax(self, z):
        """Softmax activation function with numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot_encode(self, y):
        """Convert labels to one-hot encoding."""
        n_classes = len(self.classes)
        n_samples = len(y)
        one_hot = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            class_idx = np.where(self.classes == label)[0][0]
            one_hot[i, class_idx] = 1
        return one_hot

    def _compute_cost(self, X, y_one_hot):
        """Compute cross-entropy cost with regularization."""
        X.shape[0]
        z = X.dot(self.weights) + self.bias
        predictions = self._softmax(z)


        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)


        cost = -np.mean(np.sum(y_one_hot * np.log(predictions), axis=1))


        if self.regularization == "l1":
            cost += self.reg_strength * np.sum(np.abs(self.weights))
        elif self.regularization == "l2":
            cost += self.reg_strength * np.sum(self.weights**2)

        return cost

    def _compute_gradients(self, X, y_one_hot):
        """Compute gradients for weights and bias."""
        m = X.shape[0]
        z = X.dot(self.weights) + self.bias
        predictions = self._softmax(z)

        dw = (1 / m) * X.T.dot(predictions - y_one_hot)
        db = (1 / m) * np.sum(predictions - y_one_hot, axis=0)


        if self.regularization == "l1":
            dw += self.reg_strength * np.sign(self.weights)
        elif self.regularization == "l2":
            dw += self.reg_strength * 2 * self.weights

        return dw, db

    def fit(self, X, y):
        """
        Fit the softmax regression model.

        Parameters:
        X (array-like): Training features of shape (n_samples, n_features)
        y (array-like): Target labels of shape (n_samples,)

        Returns:
        self: Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape


        self.classes = np.unique(y)
        n_classes = len(self.classes)


        self.weights = np.random.normal(0, 0.01, (n, n_classes))
        self.bias = np.zeros(n_classes)
        self.cost_history = []


        y_one_hot = self._one_hot_encode(y)

        for i in range(self.max_iter):

            cost = self._compute_cost(X, y_one_hot)
            self.cost_history.append(cost)


            dw, db = self._compute_gradients(X, y_one_hot)


            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


            if (
                i > 0
                and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance
            ):
                break

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        z = X.dot(self.weights) + self.bias
        return self._softmax(z)

    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]
