import numpy as np

from ..base_estimator import BaseEstimator
from .classifier import ClassifierMixin


class SVM(ClassifierMixin, BaseEstimator):
    """
    Support Vector Machine implementation from scratch.

    Supports linear and non-linear classification with different kernels.
    Uses gradient descent optimization for handling non-linearly separable data.
    """

    def __init__(
        self,
        kernel="linear",
        C=1.0,
        gamma="scale",
        degree=3,
        coef0=0.0,
        learning_rate=0.01,
        max_iter=1000,
        tolerance=1e-6,
        random_state=None,
    ):
        """
        Initialize SVM classifier.

        Parameters:
        kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C (float): Regularization parameter (inverse of regularization strength)
        gamma (str or float): Kernel coefficient for 'rbf', 'poly', 'sigmoid'
        degree (int): Degree for polynomial kernel
        coef0 (float): Independent term for polynomial and sigmoid kernels
        learning_rate (float): Learning rate for gradient descent
        max_iter (int): Maximum number of iterations
        tolerance (float): Tolerance for stopping criteria
        random_state (int): Random seed for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

        self.alpha = None
        self.b = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.n_support_ = None
        self.classes_ = None
        self.X_train = None

        self.loss_history = []

        if random_state is not None:
            np.random.seed(random_state)

    def _compute_gamma(self, X):
        """Compute gamma value if set to 'scale'"""
        if isinstance(self.gamma, str) and self.gamma == "scale":
            return 1.0 / (X.shape[1] * X.var())
        return self.gamma

    def _linear_kernel(self, X1, X2):
        """Linear kernel: K(x1, x2) = x1 � x2"""
        return np.dot(X1, X2.T)

    def _polynomial_kernel(self, X1, X2):
        """Polynomial kernel: K(x1, x2) = (gamma * x1 � x2 + coef0)^degree"""
        gamma = self._gamma_value
        return (gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree

    def _rbf_kernel(self, X1, X2):
        """RBF (Gaussian) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
        gamma = self._gamma_value

        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True)
        distances = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)

        return np.exp(-gamma * distances)

    def _sigmoid_kernel(self, X1, X2):
        """Sigmoid kernel: K(x1, x2) = tanh(gamma * x1 � x2 + coef0)"""
        gamma = self._gamma_value
        return np.tanh(gamma * np.dot(X1, X2.T) + self.coef0)

    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix between X1 and X2"""
        if X2 is None:
            X2 = X1

        if self.kernel == "linear":
            return self._linear_kernel(X1, X2)
        elif self.kernel == "poly":
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == "rbf":
            return self._rbf_kernel(X1, X2)
        elif self.kernel == "sigmoid":
            return self._sigmoid_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_decision_function(self, X):
        """Compute decision function values"""
        if self.support_vectors_ is None:
            raise ValueError("Model not fitted yet")

        K = self._compute_kernel_matrix(X, self.support_vectors_)

        decision = (
            np.sum(
                self.alpha[:, np.newaxis]
                * self.support_vector_labels_[:, np.newaxis]
                * K.T,
                axis=0,
            )
            + self.b
        )

        return decision

    def _compute_loss(self, X, y, alpha, b):
        """Compute SVM loss (hinge loss + regularization) - memory efficient"""
        n_samples = X.shape[0]

        batch_size = min(1000, n_samples)
        decision = np.zeros(n_samples)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]

            K_batch = self._compute_kernel_matrix(X_batch, X)
            decision[i:end_idx] = np.sum(alpha * y * K_batch, axis=1) + b

        hinge_loss = np.maximum(0, 1 - y * decision)

        regularization = 0.5 * np.sum(alpha**2)

        total_loss = np.mean(hinge_loss) + regularization / self.C

        return total_loss

    def _compute_gradients(self, X, y, alpha, b):
        """Compute gradients for alpha and b using subgradient method - memory efficient"""
        n_samples = X.shape[0]
        batch_size = min(1000, n_samples)

        decision = np.zeros(n_samples)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            K_batch = self._compute_kernel_matrix(X_batch, X)
            decision[i:end_idx] = np.sum(alpha * y * K_batch, axis=1) + b

        margin = y * decision
        violated = margin < 1

        grad_alpha = np.zeros(n_samples)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            if np.any(violated[i:end_idx]):
                grad_alpha[i:end_idx] = alpha[i:end_idx] / self.C
                grad_alpha[violated] -= y[violated] / n_samples

        grad_b = -np.sum(violated * y) / n_samples

        return grad_alpha, grad_b

    def fit(self, X, y):
        """
        Fit SVM model using gradient descent optimization.

        Parameters:
        X (array-like): Training features of shape (n_samples, n_features)
        y (array-like): Target labels of shape (n_samples,)

        Returns:
        self: Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM currently supports only binary classification")

        y_binary = np.where(y == self.classes_[0], -1, 1)

        n_samples, n_features = X.shape

        max_samples = 10000
        if n_samples > max_samples and self.kernel != "linear":
            print(
                f"Large dataset detected ({n_samples} samples). Subsampling to {max_samples} for {self.kernel} kernel."
            )
            indices = np.random.choice(n_samples, max_samples, replace=False)
            X = X[indices]
            y_binary = y_binary[indices]
            n_samples = max_samples

        self._gamma_value = self._compute_gamma(X)

        self.alpha = np.random.random(n_samples) * 0.01
        self.b = 0.0
        self.loss_history = []

        self.X_train = X.copy()

        prev_loss = float("inf")

        for _iteration in range(self.max_iter):
            loss = self._compute_loss(X, y_binary, self.alpha, self.b)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

            grad_alpha, grad_b = self._compute_gradients(
                X, y_binary, self.alpha, self.b
            )

            self.alpha -= self.learning_rate * grad_alpha
            self.b -= self.learning_rate * grad_b

            self.alpha = np.clip(self.alpha, 0, self.C)

        support_threshold = 1e-5
        support_indices = np.where(self.alpha > support_threshold)[0]

        self.support_vectors_ = X[support_indices]
        self.support_vector_labels_ = y_binary[support_indices]
        self.alpha = self.alpha[support_indices]
        self.n_support_ = len(support_indices)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        predictions (array): Predicted class labels
        """
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        decision_values = self._compute_decision_function(X)

        predictions = np.where(decision_values >= 0, self.classes_[1], self.classes_[0])

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities using Platt scaling approximation.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        probabilities (array): Probabilities of shape (n_samples, 2)
        """
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        decision_values = self._compute_decision_function(X)

        probabilities_positive = 1 / (1 + np.exp(-decision_values))
        probabilities_negative = 1 - probabilities_positive

        return np.column_stack([probabilities_negative, probabilities_positive])

    def decision_function(self, X):
        """
        Compute the decision function for samples in X.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        decision_values (array): Decision function values
        """
        return self._compute_decision_function(X)

    def get_params(self):
        """Get parameters for this estimator."""
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


class LinearSVM(SVM):
    """
    Simplified Linear SVM implementation optimized for linear kernels.
    """

    def __init__(
        self,
        C=1.0,
        learning_rate=0.01,
        max_iter=1000,
        tolerance=1e-6,
        random_state=None,
    ):
        """Initialize Linear SVM with linear kernel fixed."""
        super().__init__(
            kernel="linear",
            C=C,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tolerance=tolerance,
            random_state=random_state,
        )

        self.w = None

    def fit(self, X, y):
        """
        Fit Linear SVM using direct weight optimization.

        This is more efficient for linear kernels as we can work directly
        with the weight vector instead of storing all support vectors.
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM currently supports only binary classification")

        y_binary = np.where(y == self.classes_[0], -1, 1)

        n_samples, n_features = X.shape

        self.w = np.random.random(n_features) * 0.01
        self.b = 0.0
        self.loss_history = []

        prev_loss = float("inf")

        for _iteration in range(self.max_iter):
            decision = np.dot(X, self.w) + self.b

            margin = y_binary * decision
            loss = np.mean(np.maximum(0, 1 - margin)) + 0.5 * self.C * np.dot(
                self.w, self.w
            )
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

            violated = margin < 1

            grad_w = self.C * self.w
            grad_w -= (
                np.sum(violated[:, np.newaxis] * y_binary[:, np.newaxis] * X, axis=0)
                / n_samples
            )

            grad_b = -np.sum(violated * y_binary) / n_samples

            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

        return self

    def predict(self, X):
        """Predict using linear decision function."""
        if self.w is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        decision_values = np.dot(X, self.w) + self.b

        predictions = np.where(decision_values >= 0, self.classes_[1], self.classes_[0])

        return predictions

    def predict_proba(self, X):
        """Predict probabilities using linear decision function."""
        if self.w is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        decision_values = np.dot(X, self.w) + self.b

        probabilities_positive = 1 / (1 + np.exp(-decision_values))
        probabilities_negative = 1 - probabilities_positive

        return np.column_stack([probabilities_negative, probabilities_positive])

    def decision_function(self, X):
        """Compute decision function for linear SVM."""
        if self.w is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        return np.dot(X, self.w) + self.b
