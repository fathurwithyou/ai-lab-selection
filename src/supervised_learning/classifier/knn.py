import numpy as np

from .classifier import ClassifierMixin


class KNN(ClassifierMixin):
    """
    A K-Nearest Neighbors (KNN) classifier implementation.
    This classifier predicts the class of a sample based on the majority class of its k nearest neighbors.
    """

    def __init__(self, k=3, distance_metric="euclidean"):
        """
        Initialize the KNN classifier.
        Parameters:
        k (int): The number of neighbors to consider for classification.
        distance_metric (str): The distance metric to use ('euclidean', 'manhattan', 'minkowski').
        """
        self.k = k
        self.distance_metric = distance_metric
        self.p = 2
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the KNN classifier to the training data.

        Parameters:
        X (array-like): Training data features of shape (n_samples, n_features).
        y (array-like): Target labels of shape (n_samples,).

        Returns:
        self: Returns self for method chaining.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two data points based on the specified metric.

        Parameters:
        x1 (array-like): First data point.
        x2 (array-like): Second data point.

        Returns:
        float: Distance between the two points.
        """
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == "minkowski":
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def predict(self, X):
        """
        Predict the labels for the given data.

        Parameters:
        X (array-like): Data features to predict labels for of shape (n_samples, n_features).

        Returns:
        array-like: Predicted labels of shape (n_samples,).
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        predictions = []

        for x in X:

            distances = []
            for i, x_train in enumerate(self.X_train):
                distance = self._calculate_distance(x, x_train)
                distances.append((distance, self.y_train[i]))


            distances.sort(key=lambda x: x[0])
            k_nearest = distances[: self.k]


            neighbor_labels = [label for _, label in k_nearest]


            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)

        return np.array(predictions)
